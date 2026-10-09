// heisenbert-faceswap: real-time face swap in the browser, with live identity morphing.
// Built for heisenbert.com. PolyForm Noncommercial 1.0.0; for commercial use, email robert@chronic.ro first.
//
//   const swap = await createFaceSwap({ models, onProgress });
//   await swap.setIdentity({ image: 'https://…/character.png' });   // or { embedding } saved earlier
//   const swapped = swap.process(cameraTrack);                       // a MediaStreamTrack, ready for any <video>, recorder or WebRTC
//   swap.setIdentity({ image: other }, { morphMs: 1500 });           // morphs live

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { probe, rememberedBackend, rememberBackend, swapperPrecision, type Backend, type Capabilities } from './backend.ts';
import { DEFAULT_LANDMARKER, loadModel, type ModelSet, type ModelSpec, type Progress } from './models.ts';
import type { ToMain, ToWorker } from './protocol.ts';
import { buildMesh } from './mesh.ts';
import { FACE_OVAL } from './geometry.ts';

export type { Backend, Capabilities } from './backend.ts';
export type { ModelSet, ModelSpec, Progress } from './models.ts';
export { KNOWN_SHA256, clearModelCache } from './models.ts';
export { probe } from './backend.ts';

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
};

export type Stats = { fps: number; inferMs: number; backend: Backend };
export type IdentitySource = { embedding: Float32Array } | { image: ImageBitmapSource | string };

export type FaceSwap = {
  readonly backend: Backend;
  readonly capabilities: Capabilities;
  /** Swap to a face. Returns its 512-d identity embedding: store it to skip the identity model next time. */
  setIdentity(source: IdentitySource, opts?: { morphMs?: number }): Promise<Float32Array>;
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

  const [swapBytes, landmarkerBytes] = await Promise.all([
    loadModel('swapper', swapSpec, opts.onProgress),
    loadModel('landmarker', opts.models.landmarker ?? DEFAULT_LANDMARKER, opts.onProgress),
  ]);

  const vision = await FilesetResolver.forVisionTasks(opts.mediapipeWasm ?? MP_CDN);
  const makeLandmarker = (mode: 'VIDEO' | 'IMAGE') => FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: new Uint8Array(landmarkerBytes), delegate: 'GPU' },
    runningMode: mode, numFaces: 1, minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  }).catch(() => FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetBuffer: new Uint8Array(landmarkerBytes), delegate: 'CPU' }, runningMode: mode, numFaces: 1 }));
  const landmarker = await makeLandmarker('VIDEO');
  let imageLandmarker: FaceLandmarker | null = null;

  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'heisenbert-faceswap' });
  const pending = new Map<number, (m: ToMain) => void>();
  let seq = 0;
  let lastStats: Stats | null = null;
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
      else if (m.type === 'stats') { lastStats = m; opts.onStats?.(m); }
      else if (m.type === 'log') opts.onLog?.(m.message);
      else if (m.type === 'error' && m.id === undefined) reject(new Error(m.message));
      if ('id' in m && m.id !== undefined) { pending.get(m.id)?.(m); pending.delete(m.id); }
    };
    worker.onerror = (e) => reject(new Error(e.message || 'The face swap worker failed to start.'));
  });
  send({ type: 'init', swapper: swapBytes, backend: wanted, caps, wasmPaths: opts.ortWasmPaths ?? ORT_CDN, mesh: buildMesh(FaceLandmarker.FACE_LANDMARKS_TESSELATION, FACE_OVAL) }, [swapBytes]);
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
      const bitmap = typeof source.image === 'string'
        ? await createImageBitmap(await (await fetch(source.image)).blob())
        : await createImageBitmap(source.image);
      imageLandmarker ??= await makeLandmarker('IMAGE');
      const lm = landmarksOf(imageLandmarker.detect(bitmap));
      if (!lm) { bitmap.close(); throw new Error('No face found in that picture.'); }
      const m = await request<Extract<ToMain, { type: 'identity' }>>((id) => ({ type: 'identity', id, image: bitmap, landmarks: lm, morphMs }), [bitmap]);
      return m.embedding;
    },

    setStrength(value, ms = 400) { send({ type: 'strength', value: Math.max(0, Math.min(1, value)), ms }); },

    async processFrame(source, timestampMs = performance.now()) {
      const ts = monotonic(timestampMs);
      const lm = landmarksOf(landmarker.detectForVideo(source as HTMLVideoElement, ts));
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
        const ts = monotonic(frame.timestamp / 1000);
        let lm: Float32Array | null = null;
        try { lm = landmarksOf(landmarker.detectForVideo(frame as unknown as HTMLVideoElement, ts)); } catch { /* keep the frame unswapped */ }
        request<Extract<ToMain, { type: 'frame' }>>((id) => ({ type: 'frame', id, frame, landmarks: lm, ts }), [frame])
          // Don't wait on write(): it can stay pending while the sink is idle, which would stall the loop.
          .then((m) => { writer.write(m.frame as VideoFrame).catch(() => (m.frame as VideoFrame).close()); })
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
          ctx.transferFromImageBitmap(await api.processFrame(video));
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
