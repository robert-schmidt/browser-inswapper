/// <reference lib="webworker" />
// The swap engine, in a Web Worker so the page never stutters.
// Per frame: smooth the 5 key points → align → (when the model is free) swap the 128×128 face → paste the
// latest swapped face back through a soft face-shaped mask. Between model runs the last result is re-placed
// on the current head pose, so motion stays smooth even when the model is slower than the camera.

import * as ort from 'onnxruntime-web/all';
import { Compositor } from './compositor.ts';
import type { Mesh } from './mesh.ts';
import { candidates, sessionOptions, type Backend, type Capabilities } from './backend.ts';
import { VectorFilter } from './filters.ts';
import { FACE_OVAL, apply, keyPoints, similarity, template, toCanvas, type Affine, type Point } from './geometry.ts';
import { blendLatents, extractEmap, identityLatent } from './onnx-emap.ts';
import type { FaceMask, ToMain, ToWorker } from './protocol.ts';

const SIZE = 128;
const COLOR_MATCH = 0.3; // pull the swapped face's mean colour this far towards the original light (1 = fully: loses skin tone)

// The swap model behind one call. (ponytail: WebGPU graph capture measured within 2% of a plain run on an M2 Max,
// so it isn't used; worth re-measuring on discrete GPUs.)
type Runner = { run(target: Float32Array, source: Float32Array): Promise<Float32Array>; release(): Promise<void> };
let swapper: Runner | null = null;
let arcface: ort.InferenceSession | null = null;
let arcfaceBytes: ArrayBuffer | null = null;
let backend: Backend = 'wasm';
let caps: Capabilities;
let emap: Float32Array;

// Identity: the latent being shown, a morph in progress, and how strongly the swap is applied (fades on/off).
let latentFrom: Float32Array | null = null;
let latentTo: Float32Array | null = null;
let morphStart = 0, morphMs = 0;
let strengthFrom = 0, strengthTo = 0, strengthStart = 0, strengthMs = 0;

// Light smoothing on every landmark (pixels): steadies the still face without the lag that makes it slide.
const smooth = new VectorFilter(3.0, 0.02);
let busy = false;
const colorShifts = new Map<string, number[]>(); // per face (live: 'live'; video files: per tracked person)
let compositor: Compositor;
let stillCompositor: Compositor | null = null; // video files: separate, so a running camera swap is untouched
let mesh: Mesh;
let latestMask: FaceMask | null = null; // from the occlusion worker, a frame or two behind
const face = new OffscreenCanvas(SIZE, SIZE);
const faceCtx = face.getContext('2d')!;
const crop = new OffscreenCanvas(SIZE, SIZE);
const cropCtx = crop.getContext('2d', { willReadFrequently: true })!;
// A copy of the square around the face, taken while the frame is still here, so detection and the swap can
// run after it's gone. One per user (live swap, photo identities, video files), so they never share one.
const REGION = 384;
type Region = { canvas: OffscreenCanvas; ctx: OffscreenCanvasRenderingContext2D };
const makeRegion = (): Region => { const canvas = new OffscreenCanvas(REGION, REGION); return { canvas, ctx: canvas.getContext('2d', { willReadFrequently: true })! }; };
const liveRegion = makeRegion(), photoRegion = makeRegion(), stillRegion = makeRegion();
const maskCanvas = new OffscreenCanvas(SIZE, SIZE);
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })!;
const timings: number[] = [];
let frames = 0, lastStats = 0;

// ONNX Runtime's WebGPU backend runs one session at a time: every model run goes through this queue.
let queue: Promise<unknown> = Promise.resolve();
const exclusive = <T>(fn: () => Promise<T>): Promise<T> => { const r = queue.then(fn); queue = r.catch(() => {}); return r; };

const post = (m: ToMain, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
const ease = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
const progress = (start: number, ms: number) => (ms <= 0 ? 1 : (performance.now() - start) / ms);

function currentLatent() {
  if (!latentTo) return null;
  if (!latentFrom) return latentTo;
  const t = ease(progress(morphStart, morphMs));
  if (t >= 1) { latentFrom = null; return latentTo; }
  return blendLatents(latentFrom, latentTo, t);
}
const currentStrength = () => strengthFrom + (strengthTo - strengthFrom) * ease(progress(strengthStart, strengthMs));

async function plainRunner(bytes: ArrayBuffer, b: Backend): Promise<Runner> {
  const s = await ort.InferenceSession.create(new Uint8Array(bytes), sessionOptions(b, caps));
  return {
    run: async (target, source) => (await s.run({ target: new ort.Tensor('float32', target, [1, 3, SIZE, SIZE]), source: new ort.Tensor('float32', source, [1, 512]) })).output.data as Float32Array,
    release: () => s.release(),
  };
}

async function createSwapper(bytes: ArrayBuffer, order: Backend[]) {
  let best: { s: Runner; b: Backend; ms: number } | null = null;
  const target = new Float32Array(3 * SIZE * SIZE).fill(0.5), source = new Float32Array(512).fill(0.04);
  for (const b of order) {
    if (b === 'wasm' && best) break; // a working GPU always beats the CPU for this model
    try {
      let t = performance.now();
      const s = await plainRunner(bytes, b);
      await s.run(target, source); // first run compiles shaders
      post({ type: 'log', message: `${b}: ready in ${Math.round(performance.now() - t)} ms` });
      t = performance.now();
      for (let i = 0; i < 5; i++) await s.run(target, source);
      const ms = (performance.now() - t) / 5;
      post({ type: 'log', message: `${b}: ${ms.toFixed(1)} ms per swap` });
      if (!best || ms < best.ms) { await best?.s.release(); best = { s, b, ms }; } else await s.release();
    } catch (err) {
      post({ type: 'log', message: `${b} unavailable: ${err instanceof Error ? err.message : err}` });
    }
  }
  if (!best) throw new Error('No way to run the face model on this device.');
  return best;
}

/** Draws `source` aligned into the square crop canvas and returns its pixels. */
function alignedPixels(source: CanvasImageSource, m: Affine, size: number) {
  if (crop.width !== size) { crop.width = size; crop.height = size; }
  cropCtx.setTransform(...toCanvas(m));
  cropCtx.drawImage(source, 0, 0);
  cropCtx.setTransform(1, 0, 0, 1, 0, 0);
  return cropCtx.getImageData(0, 0, size, size);
}

/** Copies the face's surroundings (1.6× the face) into `r`; returns the tracker points in that copy's pixels. */
function snapshot(r: Region, source: CanvasImageSource, points: Float32Array) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of FACE_OVAL) { const x = points[i * 2], y = points[i * 2 + 1]; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const side = 1.6 * Math.max(x1 - x0, y1 - y0), ox = (x0 + x1 - side) / 2, oy = (y0 + y1 - side) / 2, k = REGION / side;
  r.ctx.clearRect(0, 0, REGION, REGION);
  r.ctx.drawImage(source, ox, oy, side, side, 0, 0, REGION, REGION);
  const local = new Float32Array(points.length);
  for (let i = 0; i < points.length; i += 2) { local[i] = (points[i] - ox) * k; local[i + 1] = (points[i + 1] - oy) * k; }
  return local;
}

function toTensor(img: ImageData, mean: number, std: number) {
  return new ort.Tensor('float32', toFloats(img, mean, std), [1, 3, img.height, img.width]);
}

function toFloats(img: ImageData, mean: number, std: number) {
  const n = img.width * img.height, d = img.data, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    t[i] = (d[i * 4] - mean) / std;
    t[n + i] = (d[i * 4 + 1] - mean) / std;
    t[2 * n + i] = (d[i * 4 + 2] - mean) / std;
  }
  return t;
}

/** Soft mask in crop space: the face oval (from the landmarks) feathered, inside a feathered square. */
function faceMask(ovalCrop: [number, number][]) {
  maskCtx.clearRect(0, 0, SIZE, SIZE);
  maskCtx.filter = 'blur(5px)';
  maskCtx.fillStyle = '#fff';
  maskCtx.beginPath();
  // Shrink the oval a little towards its centre: the model redraws the inner face, the hairline stays yours.
  const cx = ovalCrop.reduce((s, p) => s + p[0], 0) / ovalCrop.length, cy = ovalCrop.reduce((s, p) => s + p[1], 0) / ovalCrop.length;
  ovalCrop.forEach(([x, y], i) => { const px = cx + (x - cx) * 0.92, py = cy + (y - cy) * 0.9; if (i) maskCtx.lineTo(px, py); else maskCtx.moveTo(px, py); });
  maskCtx.closePath();
  maskCtx.fill();
  maskCtx.filter = 'none';
  maskCtx.globalCompositeOperation = 'destination-in';
  maskCtx.filter = 'blur(4px)';
  maskCtx.fillRect(10, 8, SIZE - 20, SIZE - 14);
  maskCtx.filter = 'none';
  maskCtx.globalCompositeOperation = 'source-over';
  return maskCtx.getImageData(0, 0, SIZE, SIZE).data;
}

/** Runs the swap on the current crop, colour-matches it to the original and stores it as the next patch. */
/** Runs the swap on an aligned crop and colour-matches it to the original; `key` keeps each face's colour steady over time. */
async function swapFace(original: ImageData, mask: Uint8ClampedArray, latent: Float32Array, key: string) {
  const t0 = performance.now();
  const o = await exclusive(() => swapper!.run(toFloats(original, 0, 255), latent));
  const n = SIZE * SIZE, src = original.data;
  // Colour match inside the mask: shift the swapped face's mean towards the original skin and light.
  const mo = [0, 0, 0], ms = [0, 0, 0];
  let w = 0;
  for (let i = 0; i < n; i++) {
    const a = mask[i * 4 + 3] / 255;
    if (a < 0.5) continue;
    w++;
    for (let c = 0; c < 3; c++) { mo[c] += src[i * 4 + c]; ms[c] += o[c * n + i] * 255; }
  }
  const raw = w ? [0, 1, 2].map((c) => ((mo[c] - ms[c]) / w) * COLOR_MATCH) : [0, 0, 0];
  // Smoothed over swaps, so the light on the face doesn't flicker from one result to the next.
  const prev = colorShifts.get(key);
  const shift = prev ? prev.map((v, c) => v + (raw[c] - v) * 0.4) : raw;
  colorShifts.set(key, shift);
  const img = new ImageData(SIZE, SIZE);
  const px = img.data;
  for (let i = 0; i < n; i++) {
    px[i * 4] = o[i] * 255 + shift[0];
    px[i * 4 + 1] = o[n + i] * 255 + shift[1];
    px[i * 4 + 2] = o[2 * n + i] * 255 + shift[2];
    px[i * 4 + 3] = 255;
  }
  timings.push(performance.now() - t0);
  if (timings.length > 30) timings.shift();
  return img;
}

/** Aligns and swaps a face already copied into `r` (`local`: its landmarks there): the swapped face and where each landmark sits in it. */
async function swapAt(r: Region, local: Float32Array, latent: Float32Array, key: string) {
  const m = similarity(keyPoints(local, 1, 1, 2), template(SIZE));
  const original = alignedPixels(r.canvas, m, SIZE);
  const n = local.length / 2, uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) { const [x, y] = apply(m, [local[i * 2], local[i * 2 + 1]]); uv[i * 2] = x / SIZE; uv[i * 2 + 1] = y / SIZE; }
  const oval = FACE_OVAL.map((i) => [uv[i * 2] * SIZE, uv[i * 2 + 1] * SIZE] as Point);
  return { image: await swapFace(original, faceMask(oval), latent, key), uv };
}

async function identityFromImage(bitmap: ImageBitmap, lm: Float32Array) {
  if (!arcface) {
    if (!arcfaceBytes) throw new Error('The identity model isn’t loaded.');
    arcface = await ort.InferenceSession.create(new Uint8Array(arcfaceBytes), sessionOptions(backend === 'webnn' ? 'webgpu' : backend, caps)).catch(() =>
      ort.InferenceSession.create(new Uint8Array(arcfaceBytes!), sessionOptions('wasm', caps)));
  }
  const pts = new Float32Array((lm.length / 3) * 2);
  for (let i = 0; i < pts.length / 2; i++) { pts[i * 2] = lm[i * 3] * bitmap.width; pts[i * 2 + 1] = lm[i * 3 + 1] * bitmap.height; }
  const local = snapshot(photoRegion, bitmap, pts);
  const img = alignedPixels(photoRegion.canvas, similarity(keyPoints(local, 1, 1, 2), template(112)), 112);
  const session = arcface;
  const r = await exclusive(() => session.run({ [session.inputNames[0]]: toTensor(img, 127.5, 127.5) }));
  return new Float32Array(r[session.outputNames[0]].data as Float32Array);
}

/** The live swap: on a copy taken from the frame (which has moved on by the time this finishes). */
async function swapJob(local: Float32Array, latent: Float32Array) {
  const { image, uv } = await swapAt(liveRegion, local, latent, 'live');
  faceCtx.putImageData(image, 0, 0);
  compositor.setFace(face, uv);
}

function render(frame: VideoFrame | ImageBitmap, lm: Float32Array | null, ts: number) {
  const w = 'displayWidth' in frame ? frame.displayWidth : frame.width;
  const h = 'displayHeight' in frame ? frame.displayHeight : frame.height;
  const strength = currentStrength();
  let points: Float32Array | null = null;
  if (lm) {
    const n = lm.length / 3, raw = new Array<number>(n * 2);
    for (let i = 0; i < n; i++) { raw[i * 2] = lm[i * 3] * w; raw[i * 2 + 1] = lm[i * 3 + 1] * h; }
    points = Float32Array.from(smooth.filter(raw, ts));
    // Start a swap when the model is free; frames keep flowing meanwhile, drawn with the last result.
    const latent = currentLatent();
    if (strength > 0.001 && !busy && latent && swapper) {
      busy = true;
      const local = snapshot(liveRegion, frame, points);
      swapJob(local, latent).catch((err) => post({ type: 'log', message: `swap failed: ${err}` })).finally(() => { busy = false; });
    }
  } else {
    smooth.reset();
    compositor.clearFace(); // a new face appearing shouldn't get the old one's pixels
  }
  compositor.draw(frame, w, h, points, strength, latestMask);
  frames++;
  const now = performance.now();
  if (now - lastStats > 1000) {
    post({ type: 'stats', fps: (frames * 1000) / (now - lastStats || 1), inferMs: timings.length ? timings.reduce((a, b) => a + b, 0) / timings.length : 0, backend });
    frames = 0; lastStats = now;
  }
  return compositor.canvas;
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      caps = m.caps;
      if (m.wasmPaths) ort.env.wasm.wasmPaths = m.wasmPaths;
      ort.env.wasm.numThreads = caps.wasmThreads ? Math.max(1, Math.min(8, caps.cores - 1)) : 1;
      emap = extractEmap(m.swapper);
      const order = m.backend === 'auto' ? candidates(caps) : [m.backend];
      const made = await createSwapper(m.swapper, order);
      swapper = made.s; backend = made.b;
      arcfaceBytes = m.arcface ?? null;
      mesh = m.mesh;
      compositor = new Compositor(m.mesh);
      if (m.masks) m.masks.onmessage = (ev: MessageEvent<FaceMask>) => { latestMask = ev.data; };
      post({ type: 'ready', backend, inferMs: made.ms });
    } else if (m.type === 'arcface') {
      arcfaceBytes = m.bytes;
    } else if (m.type === 'identity') {
      const embedding = m.embedding ?? (m.image && m.landmarks ? await identityFromImage(m.image, m.landmarks) : null);
      m.image?.close();
      if (!embedding) throw new Error('No face found in that picture.');
      const next = identityLatent(embedding, emap);
      latentFrom = latentTo ? currentLatent() : null;
      latentTo = next;
      morphStart = performance.now(); morphMs = latentFrom ? m.morphMs : 0;
      if (strengthTo === 0) { strengthFrom = currentStrength(); strengthTo = 1; strengthStart = performance.now(); strengthMs = m.morphMs || 300; }
      post({ type: 'identity', id: m.id, embedding }, [embedding.buffer]);
    } else if (m.type === 'embed') {
      const embeddings: Float32Array[] = [];
      for (const lm of m.faces) embeddings.push(await identityFromImage(m.image, lm));
      m.image.close();
      post({ type: 'embeddings', id: m.id, embeddings }, embeddings.map((e) => e.buffer));
    } else if (m.type === 'still') {
      const latent = currentLatent();
      const w = m.frame.width, h = m.frame.height;
      const faces = [];
      for (const f of m.faces) {
        const target = f.identity ? identityLatent(f.identity, emap) : latent;
        if (!target) continue;
        const local = snapshot(stillRegion, m.frame, f.points);
        const { image, uv } = await swapAt(stillRegion, local, target, f.key);
        faces.push({ image, uv, points: f.points });
      }
      stillCompositor ??= new Compositor(mesh);
      stillCompositor.drawStill(m.frame, w, h, faces, m.mask);
      m.frame.close();
      const bmp = stillCompositor.canvas.transferToImageBitmap();
      post({ type: 'frame', id: m.id, frame: bmp }, [bmp]);
    } else if (m.type === 'strength') {
      strengthFrom = currentStrength(); strengthTo = m.value; strengthStart = performance.now(); strengthMs = m.ms;
    } else if (m.type === 'frame') {
      const canvas = render(m.frame, m.landmarks, m.ts);
      const isVideoFrame = typeof VideoFrame !== 'undefined' && m.frame instanceof VideoFrame;
      if (isVideoFrame) {
        const vf = new VideoFrame(canvas, { timestamp: (m.frame as VideoFrame).timestamp });
        m.frame.close();
        post({ type: 'frame', id: m.id, frame: vf }, [vf]);
      } else {
        m.frame.close();
        const bmp = canvas.transferToImageBitmap();
        post({ type: 'frame', id: m.id, frame: bmp }, [bmp]);
      }
    }
  } catch (err) {
    if (m.type === 'frame') m.frame.close?.();
    post({ type: 'error', id: 'id' in m ? m.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
