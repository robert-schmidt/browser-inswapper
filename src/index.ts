// heisenbert-faceswap: real-time face swap in the browser, with live identity morphing.
// Built for heisenbert.com. PolyForm Noncommercial 1.0.0; for commercial use, email robert@chronic.ro first.
//
//   const swap = await createFaceSwap({ models, onProgress });
//   await swap.setIdentity({ image: 'https://…/character.png' });   // or { embedding } saved earlier
//   const swapped = swap.process(cameraTrack);                       // a MediaStreamTrack, ready for any <video>, recorder or WebRTC
//   swap.setIdentity({ image: other }, { morphMs: 1500 });           // morphs live

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { probe, rememberedBackend, rememberBackend, swapperPrecision, type Backend, type Capabilities } from './backend.ts';
import { DEFAULT_LANDMARKER, DEFAULT_SEGMENTER, loadModel, type ModelSet, type ModelSpec, type Progress } from './models.ts';
import type { ToMain, ToWorker } from './protocol.ts';
import { buildMesh } from './mesh.ts';
import { FACE_OVAL, keyPoints, similarity, template, toCanvas } from './geometry.ts';
import { faceIssues, type FaceIssue } from './faces.ts';

export type { Backend, Capabilities } from './backend.ts';
export type { ModelSet, ModelSpec, Progress } from './models.ts';
export { KNOWN_SHA256, clearModelCache } from './models.ts';
export { probe } from './backend.ts';
export { ISSUE_TEXT, type FaceIssue } from './faces.ts';

export type FaceSwapOptions = {
  models: ModelSet;
  /** 'auto' (default) benchmarks this device once and remembers the fastest. */
  backend?: Backend | 'auto';
  /** Where onnxruntime-web's .wasm/.mjs files are served (default: jsDelivr, pinned). */
  ortWasmPaths?: string;
  /** Where MediaPipe's tasks-vision wasm folder is served (default: jsDelivr, pinned). */
  mediapipeWasm?: string;
  onProgress?: (p: Progress) => void;
  onStats?: (s: Stats) => void;
  onLog?: (message: string) => void;
  /** Keep hands, microphones and other things in front of the face visible (default true; adds a 16 MB model). */
  occlusion?: boolean;
  /** Where face tracking runs: 'GPU' (default, faster on average) or 'CPU' (steadier timing, leaves the GPU to the swap model). */
  tracking?: 'CPU' | 'GPU';
};

/** `latencyMs`: from a camera frame arriving to its swapped copy leaving (tracking + compositing), averaged. */
export type Stats = { fps: number; inferMs: number; backend: Backend; latencyMs: number };
export type IdentitySource = { embedding: Float32Array } | { image: ImageBitmapSource | string; face?: FoundFace };

/** A face found in a photo by `findFaces`. */
export type FoundFace = {
  /** Where it is in the photo, in pixels. */
  box: { x: number; y: number; width: number; height: number };
  /** Its 478 landmarks, normalised to the photo (x, y, z). */
  landmarks: Float32Array;
  /** What would make the swapped face less accurate (see ISSUE_TEXT); empty for a good photo. */
  issues: FaceIssue[];
  /** The face cropped square, 160 px, for a picker. */
  preview: ImageBitmap;
};

export type FaceSwap = {
  readonly backend: Backend;
  readonly capabilities: Capabilities;
  /** Swap to a face. Returns its 512-d identity embedding: store it to skip the identity model next time. */
  setIdentity(source: IdentitySource, opts?: { morphMs?: number }): Promise<Float32Array>;
  /** Every face in a photo, biggest first, with quality notes and a preview: pass one to setIdentity({ image, face }). */
  findFaces(image: ImageBitmapSource | string): Promise<FoundFace[]>;
  /** Fade the swap in (1) or out (0) without losing the identity. */
  setStrength(value: number, ms?: number): void;
  /** Swapped copy of a camera track. Stop the returned track (or call dispose) when done. */
  process(track: MediaStreamTrack, opts?: { maxFps?: number }): MediaStreamTrack;
  /** One frame through the swap, for apps with their own render loop. */
  processFrame(source: CanvasImageSource | VideoFrame, timestampMs?: number): Promise<ImageBitmap>;
  dispose(): void;
};

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';

export async function createFaceSwap(opts: FaceSwapOptions): Promise<FaceSwap> {
  const caps = await probe();
  const wanted = opts.backend && opts.backend !== 'auto' ? opts.backend : rememberedBackend() ?? 'auto';
  // Model precision follows the likely backend: fp16 for GPUs with half precision, fp32 otherwise.
  const likely: Backend = wanted !== 'auto' ? wanted : caps.webgpu.available ? 'webgpu' : caps.webnn ? 'webnn' : 'wasm';
  const fp = swapperPrecision(likely, caps);
  const swapSpec = (fp === 'fp16' ? opts.models.swapperFp16 : opts.models.swapperFp32) ?? opts.models.swapperFp32 ?? opts.models.swapperFp16;
  if (!swapSpec) throw new Error('models.swapperFp16 or models.swapperFp32 is required.');

  const occlusion = opts.occlusion ?? true;
  const [swapBytes, landmarkerBytes, segmenterBytes] = await Promise.all([
    loadModel('swapper', swapSpec, opts.onProgress),
    loadModel('landmarker', opts.models.landmarker ?? DEFAULT_LANDMARKER, opts.onProgress),
    occlusion ? loadModel('segmenter', opts.models.segmenter ?? DEFAULT_SEGMENTER, opts.onProgress) : null,
  ]);

  const vision = await FilesetResolver.forVisionTasks(opts.mediapipeWasm ?? MP_CDN);
  const makeLandmarker = (mode: 'VIDEO' | 'IMAGE', numFaces = 1, delegate: 'CPU' | 'GPU' = 'GPU') => FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: new Uint8Array(landmarkerBytes), delegate },
    runningMode: mode, numFaces, minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  }).catch(() => FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetBuffer: new Uint8Array(landmarkerBytes), delegate: 'CPU' }, runningMode: mode, numFaces }));
  const landmarker = await makeLandmarker('VIDEO', 1, opts.tracking ?? 'GPU');
  let imageLandmarker: FaceLandmarker | null = null;
  let groupLandmarker: FaceLandmarker | null = null;
  // Occlusion runs in its own worker on small copies of the frame and sends masks straight to the swap worker.
  const segWorker = segmenterBytes ? new Worker(new URL('./segment-worker.ts', import.meta.url), { type: 'module', name: 'heisenbert-faceswap-occlusion' }) : null;
  const masks = new MessageChannel();
  let segIdle = false, lastSeg = 0;
  if (segWorker) {
    segWorker.onmessage = (e: MessageEvent<{ type: string; message?: string }>) => {
      if (e.data.type === 'ready' || e.data.type === 'done') segIdle = true;
      else if (e.data.type === 'error') opts.onLog?.(`occlusion off: ${e.data.message}`);
    };
    segWorker.postMessage({ type: 'init', model: segmenterBytes, wasm: opts.mediapipeWasm ?? MP_CDN, port: masks.port1 }, [segmenterBytes!, masks.port1]);
  }
  const segment = (source: CanvasImageSource | VideoFrame, w: number, h: number, ts: number) => {
    // ponytail: 8 masks/s keeps the GPU for the swap; raise on fast GPUs if hands visibly lag.
    if (!segWorker || !segIdle || !w || ts - lastSeg < 125) return;
    segIdle = false; lastSeg = ts;
    const copy = typeof VideoFrame !== 'undefined' && source instanceof VideoFrame ? source.clone() : source;
    createImageBitmap(copy as ImageBitmapSource, { resizeWidth: 256, resizeHeight: Math.max(16, Math.round((256 * h) / w)), resizeQuality: 'low' })
      .then((bitmap) => segWorker.postMessage({ type: 'frame', bitmap, ts }, [bitmap]))
      .catch(() => { segIdle = true; })
      .finally(() => { if (copy !== source) (copy as VideoFrame).close(); });
  };
  const sizeOf = (s: CanvasImageSource | VideoFrame) =>
    'displayWidth' in s ? [s.displayWidth, s.displayHeight] : 'videoWidth' in s ? [s.videoWidth, s.videoHeight] : [Number((s as HTMLCanvasElement).width), Number((s as HTMLCanvasElement).height)];

  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'heisenbert-faceswap' });
  const pending = new Map<number, (m: ToMain) => void>();
  let seq = 0;
  let lastStats: Stats | null = null;
  let latency = 0;
  const measured = (t0: number) => { const ms = performance.now() - t0; latency = latency ? latency * 0.9 + ms * 0.1 : ms; };
  const send = (m: ToWorker, transfer: Transferable[] = []) => worker.postMessage(m, transfer);
  const request = <T extends ToMain>(build: (id: number) => ToWorker, transfer: Transferable[] = []) =>
    new Promise<T>((resolve, reject) => {
      const id = ++seq;
      pending.set(id, (m) => (m.type === 'error' ? reject(new Error(m.message)) : resolve(m as T)));
      send(build(id), transfer);
    });

  const ready = new Promise<{ backend: Backend }>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<ToMain>) => {
      const m = e.data;
      if (m.type === 'ready') resolve(m);
      else if (m.type === 'stats') { lastStats = { ...m, latencyMs: Math.round(latency) }; opts.onStats?.(lastStats); }
      else if (m.type === 'log') opts.onLog?.(m.message);
      else if (m.type === 'error' && m.id === undefined) reject(new Error(m.message));
      if ('id' in m && m.id !== undefined) { pending.get(m.id)?.(m); pending.delete(m.id); }
    };
    worker.onerror = (e) => reject(new Error(e.message || 'The face swap worker failed to start.'));
  });
  send({ type: 'init', swapper: swapBytes, backend: wanted, caps, wasmPaths: opts.ortWasmPaths ?? ORT_CDN, mesh: buildMesh(FaceLandmarker.FACE_LANDMARKS_TESSELATION, FACE_OVAL), masks: segWorker ? masks.port2 : undefined },
    segWorker ? [swapBytes, masks.port2] : [swapBytes]);
  const { backend } = await ready;
  rememberBackend(backend);

  let arcfaceSent = false;
  const landmarksOf = (result: { faceLandmarks: { x: number; y: number; z: number }[][] }) => {
    const face = result.faceLandmarks[0];
    if (!face) return null;
    const a = new Float32Array(face.length * 3);
    face.forEach((p, i) => { a[i * 3] = p.x; a[i * 3 + 1] = p.y; a[i * 3 + 2] = p.z; });
    return a;
  };
  const toBitmap = async (image: ImageBitmapSource | string) =>
    typeof image === 'string' ? createImageBitmap(await (await fetch(image)).blob()) : createImageBitmap(image);
  const boxOf = (lm: Float32Array, W: number, H: number) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const i of FACE_OVAL) { x0 = Math.min(x0, lm[i * 3] * W); x1 = Math.max(x1, lm[i * 3] * W); y0 = Math.min(y0, lm[i * 3 + 1] * H); y1 = Math.max(y1, lm[i * 3 + 1] * H); }
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
  };
  let lastTs = 0;
  const monotonic = (ts: number) => (lastTs = Math.max(lastTs + 1, Math.round(ts)));

  const api: FaceSwap = {
    backend,
    capabilities: caps,

    async setIdentity(source, o = {}) {
      const morphMs = o.morphMs ?? 0;
      if ('embedding' in source) {
        const m = await request<Extract<ToMain, { type: 'identity' }>>((id) => ({ type: 'identity', id, embedding: source.embedding.slice(), morphMs }));
        return m.embedding;
      }
      if (!arcfaceSent) {
        if (!opts.models.arcface) throw new Error('models.arcface is needed to swap to a face from a picture.');
        const bytes = await loadModel('arcface', opts.models.arcface, opts.onProgress);
        send({ type: 'arcface', bytes }, [bytes]);
        arcfaceSent = true;
      }
      const bitmap = await toBitmap(source.image);
      imageLandmarker ??= await makeLandmarker('IMAGE');
      const lm = source.face?.landmarks.slice() ?? landmarksOf(imageLandmarker.detect(bitmap));
      if (!lm) { bitmap.close(); throw new Error('No face found in that picture.'); }
      const m = await request<Extract<ToMain, { type: 'identity' }>>((id) => ({ type: 'identity', id, image: bitmap, landmarks: lm, morphMs }), [bitmap]);
      return m.embedding;
    },

    async findFaces(image) {
      const bitmap = await toBitmap(image);
      try {
        groupLandmarker ??= await makeLandmarker('IMAGE', 6);
        imageLandmarker ??= await makeLandmarker('IMAGE');
        const W = bitmap.width, H = bitmap.height;
        const all = (src: ImageBitmap) => groupLandmarker!.detect(src).faceLandmarks.map((f) => Float32Array.from(f.flatMap((p) => [p.x, p.y, p.z])));
        let found = all(bitmap);
        if (!found.length) {
          // Small faces (group or full-body shots): the detector needs a face to fill about a fifth of the
          // picture, so look again in finer and finer overlapping tiles until something turns up.
          for (let n = 2; n <= 5 && !found.length; n++) {
            const size = 1.5 / n, step = (1 - size) / (n - 1);
            for (let ty = 0; ty < n; ty++) for (let tx = 0; tx < n; tx++) {
              const sx = tx * step * W, sy = ty * step * H, sw = size * W, sh = size * H;
              const tile = await createImageBitmap(bitmap, sx, sy, sw, sh);
              for (const f of all(tile)) {
                for (let i = 0; i < f.length; i += 3) { f[i] = (sx + f[i] * sw) / W; f[i + 1] = (sy + f[i + 1] * sh) / H; }
                found.push(f);
              }
              tile.close();
            }
          }
          found = found.filter((f, i) => !found.slice(0, i).some((g) => { const a = boxOf(f, W, H), b = boxOf(g, W, H); return Math.abs(a.x + a.width / 2 - b.x - b.width / 2) < b.width / 2 && Math.abs(a.y + a.height / 2 - b.y - b.height / 2) < b.height / 2; }));
        }
        const faces: FoundFace[] = [];
        for (const coarse of found) {
          // Refine on a close crop: the tracker is much more precise when the face fills the picture.
          const b0 = boxOf(coarse, W, H), side = 2 * Math.max(b0.width, b0.height);
          const sx = b0.x + b0.width / 2 - side / 2, sy = b0.y + b0.height / 2 - side / 2;
          const crop = await createImageBitmap(bitmap, sx, sy, side, side, { resizeWidth: 512, resizeHeight: 512 });
          const fine = imageLandmarker.detect(crop).faceLandmarks[0];
          crop.close();
          const lm = fine ? Float32Array.from(fine.flatMap((p) => [(sx + p.x * side) / W, (sy + p.y * side) / H, p.z])) : coarse;
          const box = boxOf(lm, W, H);
          const pts = new Float32Array((lm.length / 3) * 2);
          for (let i = 0; i < pts.length / 2; i++) { pts[i * 2] = lm[i * 3] * W; pts[i * 2 + 1] = lm[i * 3 + 1] * H; }
          const aligned = new OffscreenCanvas(112, 112), actx = aligned.getContext('2d', { willReadFrequently: true })!;
          actx.setTransform(...toCanvas(similarity(keyPoints(pts, 1, 1, 2), template(112))));
          actx.drawImage(bitmap, 0, 0);
          const px = actx.getImageData(0, 0, 112, 112).data, gray = new Uint8ClampedArray(112 * 112);
          for (let i = 0; i < gray.length; i++) gray[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
          const ps = 1.5 * Math.max(box.width, box.height);
          const preview = await createImageBitmap(bitmap, box.x + box.width / 2 - ps / 2, box.y + box.height / 2 - ps / 2, ps, ps, { resizeWidth: 160, resizeHeight: 160, resizeQuality: 'high' });
          faces.push({ box, landmarks: lm, issues: faceIssues(pts, gray, 112), preview });
        }
        return faces.sort((a, b) => b.box.width - a.box.width);
      } finally { bitmap.close(); }
    },

    setStrength(value, ms = 400) { send({ type: 'strength', value: Math.max(0, Math.min(1, value)), ms }); },

    async processFrame(source, timestampMs = performance.now()) {
      const ts = monotonic(timestampMs);
      const lm = landmarksOf(landmarker.detectForVideo(source as HTMLVideoElement, ts));
      const [sw, sh] = sizeOf(source);
      if (lm) segment(source, sw, sh, ts);
      const frame = typeof VideoFrame !== 'undefined' && source instanceof VideoFrame ? source : await createImageBitmap(source as ImageBitmapSource);
      const m = await request<Extract<ToMain, { type: 'frame' }>>((id) => ({ type: 'frame', id, frame, landmarks: lm, ts }), [frame as Transferable]);
      if (m.frame instanceof ImageBitmap) return m.frame;
      const bmp = await createImageBitmap(m.frame);
      m.frame.close();
      return bmp;
    },

    process(track, { maxFps = 30 } = {}) {
      const Processor = (globalThis as { MediaStreamTrackProcessor?: new (o: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> } }).MediaStreamTrackProcessor;
      const Generator = (globalThis as { MediaStreamTrackGenerator?: new (o: { kind: 'video' }) => MediaStreamTrack & { writable: WritableStream<VideoFrame> } }).MediaStreamTrackGenerator;
      if (Processor && Generator) return fastPath(track, Processor, Generator);
      return canvasPath(track, maxFps);
    },

    dispose() {
      stopped = true;
      for (const t of outputs) t.stop();
      worker.terminate();
      landmarker.close();
      imageLandmarker?.close();
      groupLandmarker?.close();
      segWorker?.terminate();
    },
  };

  let stopped = false;
  const outputs = new Set<MediaStreamTrack>();

  // Chrome/Edge: frames flow as VideoFrames; at most one in the worker at a time (late frames are dropped,
  // so latency never builds up).
  function fastPath(track: MediaStreamTrack, Processor: new (o: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }, Generator: new (o: { kind: 'video' }) => MediaStreamTrack & { writable: WritableStream<VideoFrame> }) {
    const reader = new Processor({ track }).readable.getReader();
    const gen = new Generator({ kind: 'video' });
    const writer = gen.writable.getWriter();
    outputs.add(gen);
    let inFlight = false;
    (async () => {
      while (!stopped && gen.readyState === 'live') {
        const { value: frame, done } = await reader.read();
        if (done || !frame) break;
        if (inFlight) { frame.close(); continue; }
        inFlight = true;
        const t0 = performance.now();
        const ts = monotonic(frame.timestamp / 1000);
        let lm: Float32Array | null = null;
        try {
          lm = landmarksOf(landmarker.detectForVideo(frame as unknown as HTMLVideoElement, ts));
          if (lm) segment(frame, frame.displayWidth, frame.displayHeight, ts);
        } catch { /* keep the frame unswapped */ }
        request<Extract<ToMain, { type: 'frame' }>>((id) => ({ type: 'frame', id, frame, landmarks: lm, ts }), [frame])
          // Don't wait on write(): it can stay pending while the sink is idle, which would stall the loop.
          .then((m) => { measured(t0); writer.write(m.frame as VideoFrame).catch(() => (m.frame as VideoFrame).close()); })
          .catch((err) => opts.onLog?.(`frame dropped: ${err.message}`))
          .finally(() => { inFlight = false; });
      }
      reader.cancel().catch(() => {});
    })();
    return gen;
  }

  // Safari/Firefox: a hidden video element feeds frames; the result is drawn to a canvas and captured.
  function canvasPath(track: MediaStreamTrack, maxFps: number) {
    const video = document.createElement('video');
    video.muted = true; video.playsInline = true;
    video.srcObject = new MediaStream([track]);
    void video.play();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('bitmaprenderer')!;
    const out = canvas.captureStream(maxFps).getVideoTracks()[0];
    outputs.add(out);
    let inFlight = false;
    const tick = async () => {
      if (stopped || out.readyState !== 'live') return;
      if (!inFlight && video.videoWidth) {
        inFlight = true;
        try {
          if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
          const t0 = performance.now();
          ctx.transferFromImageBitmap(await api.processFrame(video));
          measured(t0);
        } catch (err) { opts.onLog?.(`frame dropped: ${(err as Error).message}`); }
        inFlight = false;
      }
      if ('requestVideoFrameCallback' in video) (video as HTMLVideoElement & { requestVideoFrameCallback(cb: () => void): void }).requestVideoFrameCallback(() => void tick());
      else setTimeout(() => void tick(), 1000 / maxFps);
    };
    video.addEventListener('loadeddata', () => void tick(), { once: true });
    return out;
  }

  void lastStats;
  return api;
}
