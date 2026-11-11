import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { createFile, MP4BoxBuffer, MultiBufferStream, type Sample } from '../entries/all';
import { getFilePath } from './common';

/**
 * These tests reproduce two infinite-loop failures observed in a streaming downloader that
 * sends small Range requests and always fetches its next range starting at the position
 * returned by appendBuffer. The download must always make forward progress, terminate in a
 * bounded number of requests, download roughly the file size only, and extract exactly the
 * same samples (same count and same bytes) as feeding the whole file at once.
 */

/** Result of driving an ISOFile through a simulated range-request downloader. */
interface StreamResult {
  requests: number;
  downloadedBytes: number;
  ready: boolean;
  /** key -> hash digest of the extracted sample bytes, plus its length for clearer failures. */
  samples: Map<string, { length: number; digest: string }>;
  /** true if appendBuffer ever returned a position not strictly after the requested one. */
  noForwardProgress: boolean;
  peakBufferedBytes: number;
}

function digestSample(data: Uint8Array) {
  return {
    length: data.byteLength,
    digest: createHash('sha256').update(data).digest('hex'),
  };
}

/**
 * Simulates a downloader that requests fixed-size ranges and always continues from the
 * position returned by appendBuffer, exactly as a real range-based downloader would.
 */
function streamFile(filePath: string, chunkSize: number, keepMdatData: boolean): StreamResult {
  const file = fs.readFileSync(filePath);
  const mp4 = createFile(keepMdatData);

  let ready = false;
  const samples = new Map<string, { length: number; digest: string }>();

  mp4.onReady = info => {
    ready = true;
    for (const track of info.tracks) {
      mp4.setExtractionOptions(track.id, undefined, { nbSamples: 1_000_000 });
    }
    mp4.start();
  };
  mp4.onSamples = (id, _user, extracted: Array<Sample>) => {
    for (const sample of extracted) {
      samples.set(`${id}:${sample.number}`, digestSample(new Uint8Array(sample.data)));
    }
    mp4.releaseUsedSamples(id, extracted[extracted.length - 1].number + 1);
  };

  let nextFileStart = 0;
  let requests = 0;
  let downloadedBytes = 0;
  let noForwardProgress = false;
  let peakBufferedBytes = 0;

  while (nextFileStart < file.length) {
    const end = Math.min(nextFileStart + chunkSize, file.length);
    const buffer = MP4BoxBuffer.fromArrayBuffer(
      file.buffer.slice(nextFileStart, end),
      nextFileStart,
    );
    downloadedBytes += end - nextFileStart;
    requests++;

    const returned = mp4.appendBuffer(buffer) as number;

    // A conforming downloader relies on a strictly increasing position; going backwards or
    // returning the same position would make it request the same bytes forever.
    if (returned <= nextFileStart) {
      noForwardProgress = true;
      break;
    }
    nextFileStart = returned;

    const buffered = mp4.stream.buffers.reduce((total, b) => total + b.byteLength, 0);
    peakBufferedBytes = Math.max(peakBufferedBytes, buffered);
  }

  mp4.flush();

  return { requests, downloadedBytes, ready, samples, noForwardProgress, peakBufferedBytes };
}

/** Feeds the entire file in a single appendBuffer, the reference behaviour. */
function referenceSamples(filePath: string, keepMdatData: boolean) {
  const file = fs.readFileSync(filePath);
  const mp4 = createFile(keepMdatData);
  const samples = new Map<string, { length: number; digest: string }>();

  mp4.onReady = info => {
    for (const track of info.tracks) {
      mp4.setExtractionOptions(track.id, undefined, { nbSamples: 1_000_000 });
    }
    mp4.start();
  };
  mp4.onSamples = (id, _user, extracted: Array<Sample>) => {
    for (const sample of extracted) {
      samples.set(`${id}:${sample.number}`, digestSample(new Uint8Array(sample.data)));
    }
    mp4.releaseUsedSamples(id, extracted[extracted.length - 1].number + 1);
  };

  mp4.appendBuffer(MP4BoxBuffer.fromArrayBuffer(file.buffer.slice(0), 0));
  mp4.flush();
  return samples;
}

function expectSameSamples(
  actual: Map<string, { length: number; digest: string }>,
  expected: Map<string, { length: number; digest: string }>,
) {
  expect(actual.size).toBe(expected.size);
  for (const key of expected.keys()) {
    const actualSample = actual.get(key);
    const expectedSample = expected.get(key);
    expect(actualSample, `missing sample ${key}`).toBeDefined();
    if (!actualSample || !expectedSample) continue;
    // Compare length and a cryptographic digest: exact byte equality without copying large
    // video samples into plain arrays (which is prohibitively slow).
    expect(actualSample.length).toBe(expectedSample.length);
    expect(actualSample.digest).toBe(expectedSample.digest);
  }
}

describe('Progressive small-chunk downloading', () => {
  // The two files that used to send the downloader into an infinite loop.
  const loopingFixtures = [
    { group: 'isobmff', name: '10_fragments.mp4', expectedSamples: 470 },
    { group: 'uvvu', name: 'Solekai007_1920_29_1x1_v7clear.uvu', expectedSamples: 1202 },
  ];

  // Files that already worked; guard against regressions across request sizes.
  const otherFixtures = [
    { group: 'nalu/hevc', name: 'hvc1_only.mp4' },
    { group: 'isobmff', name: 'fragment_random_access-2.mp4' },
  ];

  const requestSizes = [65_536, 4096, 1000];

  it.for(loopingFixtures)(
    'never loops and matches a full-file feed at every request size: $name',
    async fixture => {
      const { testFile } = getFilePath(fixture.group, fixture.name);
      const fileSize = fs.statSync(testFile).size;
      const expected = referenceSamples(testFile, false);

      for (const chunkSize of requestSizes) {
        const result = streamFile(testFile, chunkSize, false);

        // The downloader must always move forward and reach the end of the file.
        expect(result.noForwardProgress).toBe(false);
        expect(result.requests).toBeLessThanOrEqual(Math.ceil(fileSize / chunkSize) + 1);
        expect(result.ready).toBe(true);

        // The amount of downloaded data must equal the file size: a loop used to download
        // more than a hundred times the file size.
        expect(result.downloadedBytes).toBe(fileSize);

        // Same number of samples and byte-identical sample content as a full-file feed.
        expect(result.samples.size).toBe(fixture.expectedSamples);
        expectSameSamples(result.samples, expected);
      }
    },
    60_000,
  );

  it.for(otherFixtures)(
    'keeps working at every request size: $name',
    async fixture => {
      const { testFile } = getFilePath(fixture.group, fixture.name);
      const fileSize = fs.statSync(testFile).size;
      const expected = referenceSamples(testFile, false);

      for (const chunkSize of requestSizes) {
        const result = streamFile(testFile, chunkSize, false);
        expect(result.noForwardProgress).toBe(false);
        expect(result.ready).toBe(true);
        expect(result.downloadedBytes).toBe(fileSize);
        expectSameSamples(result.samples, expected);
      }
    },
    30_000,
  );

  // The mdat-keeping path powers the segmentation output and must follow the same flow.
  it.for(loopingFixtures)(
    'keeps mdat data correctly with small requests (segmentation path): $name',
    async fixture => {
      const { testFile } = getFilePath(fixture.group, fixture.name);
      const fileSize = fs.statSync(testFile).size;
      const expected = referenceSamples(testFile, true);

      for (const chunkSize of [4096, 1000]) {
        const result = streamFile(testFile, chunkSize, true);
        expect(result.noForwardProgress).toBe(false);
        expect(result.ready).toBe(true);
        expect(result.downloadedBytes).toBe(fileSize);
        expect(result.samples.size).toBe(fixture.expectedSamples);
        expectSameSamples(result.samples, expected);
      }
    },
    60_000,
  );

  it('releases processed data instead of retaining the whole file (discardMdatData)', () => {
    const { testFile } = getFilePath('isobmff', '10_fragments.mp4');
    const fileSize = fs.statSync(testFile).size;
    const result = streamFile(testFile, 1000, false);

    expect(result.noForwardProgress).toBe(false);
    // Memory used by the streaming buffers must remain a small fraction of the whole file,
    // proving used data is released rather than accumulated until the end.
    expect(result.peakBufferedBytes).toBeLessThan(fileSize * 0.2);
    // All samples must still have been delivered correctly despite the aggressive cleanup.
    expect(result.samples.size).toBe(470);
  });

  it('produces correct segmentation output regardless of request size', () => {
    const { testFile } = getFilePath('isobmff', '10_fragments.mp4');
    const file = fs.readFileSync(testFile);
    const segmentedTrackIds = [201, 101];

    const runSegmentation = (chunkSize: number) => {
      const mp4 = createFile(true);
      const output = new MultiBufferStream();
      let offset = 0;
      let segmentCount = 0;

      mp4.onReady = info => {
        for (const track of info.tracks) {
          if (segmentedTrackIds.includes(track.id)) {
            mp4.setSegmentOptions(track.id, undefined, {
              nbSamples: 100,
              nbSamplesPerFragment: 50,
              rapAlignement: false,
            });
          }
        }
        const { buffer } = mp4.initializeSegmentation();
        output.insertBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, offset));
        offset += buffer.byteLength;

        mp4.onSegment = (_id, _user, segment) => {
          output.insertBuffer(MP4BoxBuffer.fromArrayBuffer(segment, offset));
          offset += segment.byteLength;
          segmentCount++;
        };
        mp4.start();
      };

      let nextFileStart = 0;
      while (nextFileStart < file.length) {
        const end = Math.min(nextFileStart + chunkSize, file.length);
        const returned = mp4.appendBuffer(
          MP4BoxBuffer.fromArrayBuffer(file.buffer.slice(nextFileStart, end), nextFileStart),
        ) as number;
        expect(returned).toBeGreaterThan(nextFileStart);
        nextFileStart = returned;
      }
      mp4.flush();

      // Re-parse the generated fragmented output and collect the actual media sample bytes.
      const reparsed = createFile(true, output);
      const sampleDigests = new Map<string, { length: number; digest: string }>();
      reparsed.onReady = info => {
        for (const track of info.tracks) {
          reparsed.setExtractionOptions(track.id, undefined, { nbSamples: 1_000_000 });
        }
        reparsed.start();
      };
      reparsed.onSamples = (id, _user, extracted: Array<Sample>) => {
        for (const sample of extracted) {
          sampleDigests.set(`${id}:${sample.number}`, digestSample(new Uint8Array(sample.data)));
        }
        reparsed.releaseUsedSamples(id, extracted[extracted.length - 1].number + 1);
      };
      reparsed.flush();

      const bytes = new Uint8Array(output.getAbsoluteEndPosition());
      for (const buffer of output.buffers) {
        bytes.set(new Uint8Array(buffer), buffer.fileStart);
      }
      return {
        segmentCount,
        outputBytes: offset,
        sampleDigests,
        digest: createHash('sha256').update(bytes).digest('hex'),
      };
    };

    const reference = runSegmentation(65_536);
    const smallChunkResults = [4096, 1000].map(chunkSize => ({
      chunkSize,
      result: runSegmentation(chunkSize),
    }));

    for (const { result } of smallChunkResults) {
      // Every media sample must survive the fragmented round-trip with identical bytes,
      // regardless of how small the download requests were.
      expect(result.sampleDigests.size).toBe(reference.sampleDigests.size);
      expectSameSamples(result.sampleDigests, reference.sampleDigests);
      expect(result.segmentCount).toBeGreaterThan(0);
      expect(result.outputBytes).toBeGreaterThan(0);
    }

    // Two different small request sizes produce byte-identical fragmented output, proving the
    // segmentation flow is stable once chunk boundaries are smaller than the segment size.
    expect(smallChunkResults[0].result.digest).toBe(smallChunkResults[1].result.digest);
    expect(smallChunkResults[0].result.segmentCount).toBe(smallChunkResults[1].result.segmentCount);
  });
});
