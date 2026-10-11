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
import { FACE_OVAL, keyPoints, similarity, template, toCanvas, yawOf } from './geometry.ts';
import { faceIssues, type FaceIssue } from './faces.ts';
import type { FaceMask, IdentityView } from './protocol.ts';
import type { SwapVideoOptions, VideoEngine, VideoInfo, VideoRange, VideoScan } from './video.ts';

export type { Backend, Capabilities } from './backend.ts';
export type { ModelSet, ModelSpec, Progress } from './models.ts';
export { KNOWN_SHA256, clearModelCache } from './models.ts';
export { probe } from './backend.ts';
export { ISSUE_TEXT, type FaceIssue } from './faces.ts';
export type { IdentityView } from './protocol.ts';
export type { SwapVideoOptions, VideoInfo, VideoPerson, VideoProgress, VideoRange, VideoScan } from './video.ts';

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
  /**
   * Where the worker files are served, if your bundler mishandles workers from packages (Next.js does: it
   * assumes `window` exists). Copy dist/worker.js and dist/segment-worker.js to your public folder and pass
   * their URLs. By default they load next to this file via `new URL(…, import.meta.url)`.
   */
  workers?: { swap?: string | URL; occlusion?: string | URL };
  /** Where face tracking runs: 'GPU' (default, faster on average) or 'CPU' (steadier timing, leaves the GPU to the swap model). */
  tracking?: 'CPU' | 'GPU';
};

/** `latencyMs`: from a camera frame arriving to its swapped copy leaving (tracking + compositing), averaged. */
export type Stats = { fps: number; inferMs: number; backend: Backend; latencyMs: number };
/**
 * Who to swap to: a photo (`face` from findFaces picks one of several), a stored `embedding`, or `views`: the same
 * person photographed at several head turns (from `identityView`), so faces turned to the side use the matching photo.
 */
export type IdentitySource = { embedding: Float32Array } | { image: ImageBitmapSource | string; face?: FoundFace } | { views: IdentityView[] };

/** A face found in a photo by `findFaces`. */
export type FoundFace = {
  /** Where it is in the photo, in pixels. */
  box: { x: number; y: number; width: number; height: number };
  /** Its 478 landmarks, normalised to the photo (x, y, z). */
  landmarks: Float32Array;
  /** How far the head is turned, in degrees: 0 facing the camera, positive with the nose towards the photo's right. */
  yaw: number;
  /** What would make the swapped face less accurate (see ISSUE_TEXT); empty for a good photo. */
  issues: FaceIssue[];
  /** The face cropped square, 160 px, for a picker. */
  preview: ImageBitmap;
};

export type FaceSwap = {
  readonly backend: Backend;
  readonly capabilities: Capabilities;
  /** Swap to a face. Returns its 512-d identity embedding (with `views`, the most frontal one's): store it to skip the identity model next time. */
  setIdentity(source: IdentitySource, opts?: { morphMs?: number }): Promise<Float32Array>;
  /** Every face in a photo, biggest first, with quality notes and a preview: pass one to setIdentity({ image, face }). */
  findFaces(image: ImageBitmapSource | string): Promise<FoundFace[]>;
  /** Video files: the people in a clip (sampled across it), most seen first, to choose who to replace. */
  /** Video files: length, size and frame rate, read from the file (works for files the browser can't play). */
  videoInfo(file: Blob): Promise<VideoInfo>;
  scanVideo(file: Blob, opts?: VideoRange & { samples?: number; signal?: AbortSignal }): Promise<VideoScan>;
  /** A face's identity (512 numbers) without switching to it: for `swapVideo`'s `map`, or to store. */
  identityOf(source: { image: ImageBitmapSource | string; face?: FoundFace }): Promise<Float32Array>;
  /**
   * A face's identity plus the head turn it was photographed at. Several of one person (facing the camera, turned
   * left, turned right) passed as `{ views }` to setIdentity or swapVideo's `map` give side-on faces the identity
   * measured from the side. Storable like an embedding.
   */
  identityView(source: { image: ImageBitmapSource | string; face?: FoundFace }): Promise<IdentityView>;
  /** Video files: an MP4 with faces swapped to the current identity on every frame (audio kept). Slower than real time. */
  swapVideo(file: Blob, opts?: SwapVideoOptions): Promise<Blob>;
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
    runningMode: mode, numFaces, minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5, outputFacialTransformationMatrixes: true,
  }).catch(() => FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetBuffer: new Uint8Array(landmarkerBytes), delegate: 'CPU' }, runningMode: mode, numFaces, outputFacialTransformationMatrixes: true }));
  const landmarker = await makeLandmarker('VIDEO', 1, opts.tracking ?? 'GPU');
  let imageLandmarker: FaceLandmarker | null = null;
  let groupLandmarker: FaceLandmarker | null = null;
  // Occlusion runs in its own worker on small copies of the frame and sends masks straight to the swap worker.
  const segmenterCopy = segmenterBytes?.slice(0) ?? null; // for video files' own occlusion worker
  // Written out in full: bundlers only recognise `new Worker(new URL('./x', import.meta.url))` as it is.
  const occlusionWorker = (name: string) => opts.workers?.occlusion
    ? new Worker(opts.workers.occlusion, { type: 'module', name })
    : new Worker(new URL('./segment-worker.ts', import.meta.url), { type: 'module', name });
  const segWorker = segmenterBytes ? occlusionWorker('heisenbert-faceswap-occlusion') : null;
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

  const worker = opts.workers?.swap
    ? new Worker(opts.workers.swap, { type: 'module', name: 'heisenbert-faceswap' })
    : new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'heisenbert-faceswap' });
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
  // After start: a crashed worker fails what's waiting on it instead of leaving it pending forever.
  ready.then(() => {
    worker.onerror = (e) => {
      for (const settle of pending.values()) settle({ type: 'error', message: `The face swap worker stopped: ${e.message || 'unknown error'}` });
      pending.clear();
    };
  }, () => {});
  send({ type: 'init', swapper: swapBytes, backend: wanted, caps, wasmPaths: opts.ortWasmPaths ?? ORT_CDN, mesh: buildMesh(FaceLandmarker.FACE_LANDMARKS_TESSELATION, FACE_OVAL), masks: segWorker ? masks.port2 : undefined },
    segWorker ? [swapBytes, masks.port2] : [swapBytes]);
  const { backend } = await ready;
  if (wanted === 'auto') rememberBackend(backend); // a forced choice isn't a benchmark result

  let arcfaceSent = false;
  type Detection = { faceLandmarks: { x: number; y: number; z: number }[][]; facialTransformationMatrixes?: { data: ArrayLike<number> }[] };
  const landmarksOf = (result: Detection) => {
    const face = result.faceLandmarks[0];
    if (!face) return null;
    const a = new Float32Array(face.length * 3);
    face.forEach((p, i) => { a[i * 3] = p.x; a[i * 3 + 1] = p.y; a[i * 3 + 2] = p.z; });
    return a;
  };
  const yawAt = (result: Detection, i = 0) => { const m = result.facialTransformationMatrixes?.[i]; return m ? yawOf(m.data) : 0; };
  /** Every face in a detection: landmarks and head turn. */
  const facesOf = (result: Detection) => result.faceLandmarks.map((f, i) => ({ lm: Float32Array.from(f.flatMap((p) => [p.x, p.y, p.z])), yaw: yawAt(result, i) }));
  const toBitmap = async (image: ImageBitmapSource | string) =>
    typeof image === 'string' ? createImageBitmap(await (await fetch(image)).blob()) : createImageBitmap(image);
  const boxOf = (lm: Float32Array, W: number, H: number) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const i of FACE_OVAL) { x0 = Math.min(x0, lm[i * 3] * W); x1 = Math.max(x1, lm[i * 3] * W); y0 = Math.min(y0, lm[i * 3 + 1] * H); y1 = Math.max(y1, lm[i * 3 + 1] * H); }
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
  };
  const videoEngine: VideoEngine = {
    findFaces: (image) => api.findFaces(image),
    async embed(image, faces) {
      if (!arcfaceSent) {
        if (!opts.models.arcface) throw new Error('models.arcface is needed to tell people apart in a video.');
        const bytes = await loadModel('arcface', opts.models.arcface, opts.onProgress);
        send({ type: 'arcface', bytes }, [bytes]);
        arcfaceSent = true;
      }
      const m = await request<Extract<ToMain, { type: 'embeddings' }>>((id) => ({ type: 'embed', id, image, faces }), [image]);
      return m.embeddings;
    },
    async tracker() {
      const t = await makeLandmarker('VIDEO', 6);
      return {
        detect: (frame, ts) => facesOf(t.detectForVideo(frame as unknown as HTMLVideoElement, ts)),
        close: () => t.close(),
      };
    },
    occlusion() {
      if (!segmenterCopy) return null;
      const w = occlusionWorker('heisenbert-faceswap-video-occlusion');
      const ready = new Promise<void>((resolve) => { w.onmessage = (e) => { if (e.data.type === 'ready' || e.data.type === 'error') resolve(); }; });
      w.postMessage({ type: 'init', model: segmenterCopy.slice(0), wasm: opts.mediapipeWasm ?? MP_CDN });
      return {
        async mask(small, ts) {
          await ready;
          return new Promise<FaceMask | null>((resolve) => {
            let mask: FaceMask | null = null;
            // A segmenter that stops answering (lost GPU context, crashed worker) costs this frame its mask, not the video.
            const late = setTimeout(() => resolve(null), 5000);
            w.onmessage = (e) => { if (e.data.type === 'mask') mask = e.data.mask; else if (e.data.type === 'done') { clearTimeout(late); resolve(mask); } };
            w.postMessage({ type: 'frame', bitmap: small, ts, reply: true }, [small]);
          });
        },
        close: () => w.terminate(),
      };
    },
    async still(frame, faces, mask) {
      const m = await request<Extract<ToMain, { type: 'frame' }>>((id) => ({ type: 'still', id, frame, faces, mask }),
        [frame, ...faces.flatMap((f) => [f.points.buffer, ...(f.identity ?? []).map((v) => v.embedding.buffer)]), ...(mask ? [mask.data.buffer] : [])]);
      return m.frame as ImageBitmap;
    },
  };
  let lastTs = 0;
  const monotonic = (ts: number) => (lastTs = Math.max(lastTs + 1, Math.round(ts)));

  const api: FaceSwap = {
    backend,
    capabilities: caps,

    async setIdentity(source, o = {}) {
      const morphMs = o.morphMs ?? 0;
      if ('embedding' in source || 'views' in source) {
        const views = 'views' in source ? source.views.map((v) => ({ embedding: v.embedding.slice(), yaw: v.yaw })) : undefined;
        const m = await request<Extract<ToMain, { type: 'identity' }>>((id) => ({ type: 'identity', id, views, embedding: 'embedding' in source ? source.embedding.slice() : undefined, morphMs }));
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
        const all = (src: ImageBitmap) => facesOf(groupLandmarker!.detect(src));
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
                for (let i = 0; i < f.lm.length; i += 3) { f.lm[i] = (sx + f.lm[i] * sw) / W; f.lm[i + 1] = (sy + f.lm[i + 1] * sh) / H; }
                found.push(f);
              }
              tile.close();
            }
          }
          found = found.filter((f, i) => !found.slice(0, i).some((g) => { const a = boxOf(f.lm, W, H), b = boxOf(g.lm, W, H); return Math.abs(a.x + a.width / 2 - b.x - b.width / 2) < b.width / 2 && Math.abs(a.y + a.height / 2 - b.y - b.height / 2) < b.height / 2; }));
        }
        const faces: FoundFace[] = [];
        for (const coarse of found) {
          // Refine on a close crop: the tracker is much more precise when the face fills the picture.
          const b0 = boxOf(coarse.lm, W, H), side = 2 * Math.max(b0.width, b0.height);
          const sx = b0.x + b0.width / 2 - side / 2, sy = b0.y + b0.height / 2 - side / 2;
          const crop = await createImageBitmap(bitmap, sx, sy, side, side, { resizeWidth: 512, resizeHeight: 512 });
          const close = imageLandmarker.detect(crop), fine = close.faceLandmarks[0];
          crop.close();
          const lm = fine ? Float32Array.from(fine.flatMap((p) => [(sx + p.x * side) / W, (sy + p.y * side) / H, p.z])) : coarse.lm;
          const yaw = fine ? yawAt(close) : coarse.yaw;
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
          faces.push({ box, landmarks: lm, yaw, issues: faceIssues(pts, gray, 112), preview });
        }
        return faces.sort((a, b) => b.box.width - a.box.width);
      } finally { bitmap.close(); }
    },

    identityOf: async (source) => (await api.identityView(source)).embedding,

    async identityView(source) {
      const bitmap = await toBitmap(source.image);
      imageLandmarker ??= await makeLandmarker('IMAGE');
      const detected = source.face ? null : imageLandmarker.detect(bitmap);
      const lm = source.face?.landmarks.slice() ?? landmarksOf(detected!);
      if (!lm) { bitmap.close(); throw new Error('No face found in that picture.'); }
      const yaw = source.face?.yaw ?? yawAt(detected!);
      const [embedding] = await videoEngine.embed(bitmap, [lm]);
      return { embedding, yaw };
    },

    videoInfo: async (file) => (await import('./video.ts')).videoInfo(file),
    scanVideo: async (file, o) => (await import('./video.ts')).scanVideo(videoEngine, file, o),
    swapVideo: async (file, o) => (await import('./video.ts')).swapVideo(videoEngine, file, o),

    setStrength(value, ms = 400) { send({ type: 'strength', value: Math.max(0, Math.min(1, value)), ms }); },

    async processFrame(source, timestampMs = performance.now()) {
      const ts = monotonic(timestampMs);
      const result = landmarker.detectForVideo(source as HTMLVideoElement, ts), lm = landmarksOf(result), yaw = yawAt(result);
      const [sw, sh] = sizeOf(source);
      if (lm) segment(source, sw, sh, ts);
      const frame = typeof VideoFrame !== 'undefined' && source instanceof VideoFrame ? source : await createImageBitmap(source as ImageBitmapSource);
      const m = await request<Extract<ToMain, { type: 'frame' }>>((id) => ({ type: 'frame', id, frame, landmarks: lm, yaw, ts }), [frame as Transferable]);
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
        let lm: Float32Array | null = null, yaw = 0;
        try {
          const result = landmarker.detectForVideo(frame as unknown as HTMLVideoElement, ts);
          lm = landmarksOf(result); yaw = yawAt(result);
          if (lm) segment(frame, frame.displayWidth, frame.displayHeight, ts);
        } catch { /* keep the frame unswapped */ }
        request<Extract<ToMain, { type: 'frame' }>>((id) => ({ type: 'frame', id, frame, landmarks: lm, yaw, ts }), [frame])
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
