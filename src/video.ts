// Video files: find the people in a clip, then swap chosen faces on every frame, all in the browser.
// Reading and writing the file is Mediabunny (WebCodecs: the browser's hardware decoder and encoder), loaded only
// when a video is used. Unlike the camera path, every frame gets a fresh swap for every face (no reuse of older
// results), so the output is as good as the model gets; it runs slower than real time on most devices.

import { VectorFilter } from './filters.ts';
import { FACE_OVAL } from './geometry.ts';
import type { FaceMask } from './protocol.ts';
import type { FoundFace } from './index.ts';

/** A person seen in a clip by `scanVideo`. */
export type VideoPerson = {
  id: number;
  /** Their face, cropped square (160 px), from the frame where it was biggest. */
  preview: ImageBitmap;
  /** Average identity (512 numbers), used to recognise them on every frame. */
  embedding: Float32Array;
  /** In how many of the sampled frames they appear. */
  seen: number;
};

export type VideoScan = { people: VideoPerson[]; duration: number; width: number; height: number; fps: number };
export type VideoProgress = { progress: number; etaSeconds: number | null; frames: number };
export type SwapVideoOptions = {
  /** People from `scanVideo`. With `swap`, only faces recognised as those people are replaced. */
  people?: VideoPerson[];
  /** Ids of the people to replace. Leave out to replace every face. */
  swap?: number[];
  onProgress?: (p: VideoProgress) => void;
  signal?: AbortSignal;
};

/** What the engine (index.ts) lends the video pipeline. */
export type VideoEngine = {
  findFaces(image: ImageBitmap): Promise<FoundFace[]>;
  /** Identity embeddings for faces (landmarks normalised to `image`). Takes ownership of `image`. */
  embed(image: ImageBitmap, faces: Float32Array[]): Promise<Float32Array[]>;
  /** A multi-face tracker for one pass over a clip. */
  tracker(): Promise<{ detect(frame: VideoFrame, ts: number): Float32Array[]; close(): void }>;
  /** Hand and object masks for one pass, or null when occlusion is off. Takes ownership of each small bitmap. */
  occlusion(): { mask(small: ImageBitmap, ts: number): Promise<FaceMask | null>; close(): void } | null;
  /** The frame with each face swapped (points: pixel x/y pairs). Takes ownership of `frame`. */
  still(frame: ImageBitmap, faces: { key: string; points: Float32Array }[], mask: FaceMask | null): Promise<ImageBitmap>;
};

const SAME_PERSON = 0.45; // ArcFace cosine between two sightings of one person (different people sit near 0–0.2)
const RECHECK_FRAMES = 15; // re-confirm who a tracked face is this often

const cosine = (a: Float32Array, b: Float32Array) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb);
};

const bounds = (lm: Float32Array, w: number, h: number) => {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of FACE_OVAL) { x0 = Math.min(x0, lm[i * 3] * w); x1 = Math.max(x1, lm[i * 3] * w); y0 = Math.min(y0, lm[i * 3 + 1] * h); y1 = Math.max(y1, lm[i * 3 + 1] * h); }
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, size: Math.max(x1 - x0, y1 - y0) };
};

async function open(file: Blob) {
  const mb = await import('mediabunny');
  const input = new mb.Input({ formats: mb.ALL_FORMATS, source: new mb.BlobSource(file) });
  const track = await input.getPrimaryVideoTrack().catch(() => null);
  if (!track) { input.dispose(); throw new Error('This file has no video we can read. Try an MP4, MOV or WebM.'); }
  return { mb, input, track };
}

/** Samples frames across the clip and groups the faces it finds into people, most seen first. */
export async function scanVideo(engine: VideoEngine, file: Blob, opts: { samples?: number; signal?: AbortSignal } = {}): Promise<VideoScan> {
  const { mb, input, track } = await open(file);
  try {
    const duration = await track.computeDuration();
    const fps = (await track.computePacketStats(120)).averagePacketRate;
    const sink = new mb.VideoSampleSink(track);
    const samples = opts.samples ?? 12;
    const people: (VideoPerson & { size: number })[] = [];
    for (let k = 0; k < samples; k++) {
      opts.signal?.throwIfAborted();
      const sample = await sink.getSample((duration * (k + 0.5)) / samples);
      if (!sample) continue;
      const frame = sample.toVideoFrame();
      sample.close();
      const bitmap = await createImageBitmap(frame);
      frame.close();
      const faces = await engine.findFaces(bitmap);
      const embeddings = faces.length ? await engine.embed(await createImageBitmap(bitmap), faces.map((f) => f.landmarks)) : [];
      bitmap.close();
      faces.forEach((f, i) => {
        const e = embeddings[i];
        const size = f.box.width * f.box.height;
        let best: (typeof people)[number] | null = null, bestCos = SAME_PERSON;
        for (const p of people) { const c = cosine(p.embedding, e); if (c > bestCos) { best = p; bestCos = c; } }
        if (!best) { people.push({ id: people.length, preview: f.preview, embedding: e.slice(), seen: 1, size }); return; }
        // Running mean of the identity; the biggest sighting gives the preview.
        for (let j = 0; j < e.length; j++) best.embedding[j] = (best.embedding[j] * best.seen + e[j]) / (best.seen + 1);
        best.seen++;
        if (size > best.size) { best.preview.close(); best.preview = f.preview; best.size = size; } else f.preview.close();
      });
    }
    people.sort((a, b) => b.seen - a.seen);
    return { people: people.map(({ size: _, ...p }) => p), duration, width: track.displayWidth, height: track.displayHeight, fps };
  } finally { input.dispose(); }
}

/** Swaps faces on every frame and returns an MP4 (original audio kept). The current identity is used. */
export async function swapVideo(engine: VideoEngine, file: Blob, opts: SwapVideoOptions = {}): Promise<Blob> {
  const { mb, input, track } = await open(file);
  const people = opts.people ?? [];
  const only = opts.swap && people.length && opts.swap.length < people.length ? new Set(opts.swap) : null;
  const tracker = await engine.tracker();
  const occlusion = engine.occlusion();
  type Track = { id: number; cx: number; cy: number; size: number; filter: VectorFilter; person: number | null; checkedAt: number };
  let tracks: Track[] = [], nextTrack = 0, frames = 0, lastTs = -1;
  const canvas = new OffscreenCanvas(2, 2), ctx = canvas.getContext('2d')!;
  try {
    const output = new mb.Output({ format: new mb.Mp4OutputFormat({ fastStart: 'in-memory' }), target: new mb.BufferTarget() });
    const conversion = await mb.Conversion.init({
      input, output, tracks: 'primary',
      video: {
        forceTranscode: true,
        allowTransformationMetadata: false, // rotate phone videos into the pixels: the tracker needs upright faces
        quality: mb.QUALITY_HIGH,
        process: async (sample) => {
          const ts = Math.max(lastTs + 1, Math.round(sample.timestamp * 1000)); lastTs = ts;
          const frame = sample.toVideoFrame();
          const w = frame.displayWidth, h = frame.displayHeight;
          let found: Float32Array[] = [];
          try { found = tracker.detect(frame, ts); } catch { /* no faces this frame */ }
          const bitmap = await createImageBitmap(frame);
          frame.close();

          // Follow each face from frame to frame (nearest centre), so smoothing and identity carry over.
          const next: { t: Track; lm: Float32Array }[] = [];
          for (const lm of found) {
            const b = bounds(lm, w, h);
            let t = tracks.find((x) => !next.some((n) => n.t === x) && Math.hypot(x.cx - b.cx, x.cy - b.cy) < x.size * 0.6);
            t ??= { id: nextTrack++, cx: 0, cy: 0, size: 0, filter: new VectorFilter(3.0, 0.02), person: null, checkedAt: -RECHECK_FRAMES };
            Object.assign(t, b);
            next.push({ t, lm });
          }
          tracks = next.map((n) => n.t);

          // Who is who, only when some people are to be left alone.
          if (only) {
            const due = next.filter((n) => frames - n.t.checkedAt >= RECHECK_FRAMES);
            if (due.length) {
              const embeddings = await engine.embed(await createImageBitmap(bitmap), due.map((n) => n.lm));
              due.forEach((n, i) => {
                let best: number | null = null, bestCos = 0.35;
                for (const p of people) { const c = cosine(p.embedding, embeddings[i]); if (c > bestCos) { best = p.id; bestCos = c; } }
                n.t.person = best; n.t.checkedAt = frames;
              });
            }
          }
          const chosen = next.filter((n) => !only || (n.t.person !== null && only.has(n.t.person)));
          const faces = chosen.map(({ t, lm }) => {
            const raw = new Array<number>((lm.length / 3) * 2);
            for (let i = 0; i < raw.length / 2; i++) { raw[i * 2] = lm[i * 3] * w; raw[i * 2 + 1] = lm[i * 3 + 1] * h; }
            return { key: `video-${t.id}`, points: Float32Array.from(t.filter.filter(raw, ts)) };
          });
          const mask = occlusion && faces.length
            ? await occlusion.mask(await createImageBitmap(bitmap, { resizeWidth: 256, resizeHeight: Math.max(16, Math.round((256 * h) / w)), resizeQuality: 'low' }), ts)
            : null;
          const out = await engine.still(bitmap, faces, mask);
          frames++;
          // A plain 2D canvas copy: Chrome won't make a VideoFrame straight from a WebGL canvas's bitmap.
          if (canvas.width !== out.width || canvas.height !== out.height) { canvas.width = out.width; canvas.height = out.height; }
          ctx.drawImage(out, 0, 0);
          out.close();
          return new mb.VideoSample(canvas, { timestamp: sample.timestamp, duration: sample.duration });
        },
      },
    });
    if (!conversion.isValid) throw new Error(`This video can’t be converted here (${conversion.discardedTracks.map((d) => d.reason).join(', ') || 'unsupported format'}).`);
    const started = performance.now();
    conversion.onProgress = (progress) => {
      const elapsed = (performance.now() - started) / 1000;
      opts.onProgress?.({ progress, frames, etaSeconds: progress > 0.02 ? (elapsed * (1 - progress)) / progress : null });
    };
    const abort = () => void conversion.cancel();
    opts.signal?.addEventListener('abort', abort, { once: true });
    try { await conversion.execute(); } finally { opts.signal?.removeEventListener('abort', abort); }
    opts.signal?.throwIfAborted();
    return new Blob([(output.target as InstanceType<typeof mb.BufferTarget>).buffer!], { type: 'video/mp4' });
  } finally {
    tracker.close();
    occlusion?.close();
    input.dispose();
  }
}
