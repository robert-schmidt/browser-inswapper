/// <reference lib="webworker" />
// The swap engine, in a Web Worker so the page never stutters.
// Per frame: smooth the 5 key points → align → (when the model is free) swap the 128×128 face → paste the
// latest swapped face back through a soft face-shaped mask. Between model runs the last result is re-placed
// on the current head pose, so motion stays smooth even when the model is slower than the camera.

import * as ort from 'onnxruntime-web/all';
import { Compositor } from './compositor.ts';
import { visibility, type Mesh } from './mesh.ts';
import { candidates, sessionOptions, type Backend, type Capabilities } from './backend.ts';
import { FaceFilter } from './filters.ts';
import { FACE_OVAL, apply, invert, keyPoints, similarity, template, toCanvas, type Affine } from './geometry.ts';
import { blendLatents, extractEmap, identityLatent, latentAt, type PoseLatent } from './onnx-emap.ts';
import type { FaceMask, IdentityView, ToMain, ToWorker } from './protocol.ts';

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

// Identity: the one being shown (one latent per photo angle), a morph in progress, and how strongly the swap is
// applied (fades on/off).
let latentFrom: PoseLatent[] | null = null;
let latentTo: PoseLatent[] | null = null;
let liveYaw = 0; // the live face's head turn, smoothed
let lastLive: { points: Float32Array; ts: number } | null = null; // the face's last tracked position
const HOLD_MS = 200; // how long a lost face stays where it was (longer: a face that left lingers)
let morphStart = 0, morphMs = 0;
let strengthFrom = 0, strengthTo = 0, strengthStart = 0, strengthMs = 0;

// Landmark smoothing (pixels): steadies the still face, follows a moving head without trailing it.
const smooth = new FaceFilter();
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
// `ox`, `oy`, `k`: where the copy came from in the frame (frame pixel = copy pixel / k + origin).
type Region = { canvas: OffscreenCanvas; ctx: OffscreenCanvasRenderingContext2D; ox: number; oy: number; k: number };
const makeRegion = (): Region => { const canvas = new OffscreenCanvas(REGION, REGION); return { canvas, ctx: canvas.getContext('2d', { willReadFrequently: true })!, ox: 0, oy: 0, k: 1 }; };
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

/** The identity for a face turned `yaw` degrees, mid-morph if one is running. */
function currentLatent(yaw: number) {
  if (!latentTo) return null;
  const to = latentAt(latentTo, yaw);
  if (!latentFrom) return to;
  const t = ease(progress(morphStart, morphMs));
  if (t >= 1) { latentFrom = null; return to; }
  return blendLatents(latentAt(latentFrom, yaw), to, t);
}
/** Photo identities → swap latents, sorted by angle. */
const poseLatents = (views: IdentityView[]): PoseLatent[] =>
  views.map((v) => ({ latent: identityLatent(v.embedding, emap), yaw: v.yaw })).sort((a, b) => a.yaw - b.yaw);
const YAW_FOLLOW = 0.3; // how quickly the identity follows a turning head (per frame): steady, without lagging the turn
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
  r.ox = ox; r.oy = oy; r.k = k;
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

/**
 * Soft mask in crop space for the colour match: the inner face (the mesh minus its outer rings, so the hairline
 * stays out), only where it faces the camera. On a turned head the outline crosses over itself; this doesn't.
 */
function faceMask(uv: Float32Array) {
  const px = uv.map((v) => v * SIZE);
  const { away } = visibility(mesh, px);
  maskCtx.clearRect(0, 0, SIZE, SIZE);
  maskCtx.filter = 'blur(3px)';
  maskCtx.fillStyle = '#fff';
  maskCtx.beginPath();
  const t = mesh.triangles;
  for (let i = 0; i < away.length; i++) {
    const [a, b, c] = [t[i * 3], t[i * 3 + 1], t[i * 3 + 2]];
    if (away[i] || mesh.weights[a] < 1 || mesh.weights[b] < 1 || mesh.weights[c] < 1) continue;
    maskCtx.moveTo(px[a * 2], px[a * 2 + 1]); maskCtx.lineTo(px[b * 2], px[b * 2 + 1]); maskCtx.lineTo(px[c * 2], px[c * 2 + 1]); maskCtx.closePath();
  }
  maskCtx.fill('nonzero');
  maskCtx.filter = 'none';
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
/**
 * Aligns and swaps a face already copied into `r` (`local`: its landmarks there): the swapped face and where each
 * landmark sits in it. `occlusion`: what's in front of the face in the frame (`w`×`h`), left out of the colour
 * match so a hand or sleeve doesn't tint the face.
 */
async function swapAt(r: Region, local: Float32Array, latent: Float32Array, key: string, occlusion: { mask: FaceMask; w: number; h: number } | null = null) {
  const m = similarity(keyPoints(local, 1, 1, 2), template(SIZE));
  const original = alignedPixels(r.canvas, m, SIZE);
  const n = local.length / 2, uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) { const [x, y] = apply(m, [local[i * 2], local[i * 2 + 1]]); uv[i * 2] = x / SIZE; uv[i * 2 + 1] = y / SIZE; }
  const mask = faceMask(uv);
  if (occlusion) {
    const { mask: o, w, h } = occlusion, back = invert(m), sx = o.width / w, sy = o.height / h;
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const [rx, ry] = apply(back, [x + 0.5, y + 0.5]);
      const mx = Math.floor((rx / r.k + r.ox) * sx), my = Math.floor((ry / r.k + r.oy) * sy);
      if (mx >= 0 && my >= 0 && mx < o.width && my < o.height && o.data[my * o.width + mx] < 128) mask[(y * SIZE + x) * 4 + 3] = 0;
    }
  }
  return { image: await swapFace(original, mask, latent, key), uv };
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
async function swapJob(local: Float32Array, latent: Float32Array, w: number, h: number) {
  const { image, uv } = await swapAt(liveRegion, local, latent, 'live', latestMask && { mask: latestMask, w, h });
  faceCtx.putImageData(image, 0, 0);
  compositor.setFace(face, uv);
}

function render(frame: VideoFrame | ImageBitmap, lm: Float32Array | null, yaw: number, ts: number) {
  const w = 'displayWidth' in frame ? frame.displayWidth : frame.width;
  const h = 'displayHeight' in frame ? frame.displayHeight : frame.height;
  const strength = currentStrength();
  let points: Float32Array | null = null;
  // A hand over the face can make tracking drop out for a frame or two: keep the face where it was meanwhile,
  // rather than flashing the original face through.
  if (!lm && lastLive && ts - lastLive.ts < HOLD_MS) points = lastLive.points;
  else if (lm) {
    const n = lm.length / 3, raw = new Array<number>(n * 2);
    for (let i = 0; i < n; i++) { raw[i * 2] = lm[i * 3] * w; raw[i * 2 + 1] = lm[i * 3 + 1] * h; }
    points = Float32Array.from(smooth.filter(raw, ts));
    lastLive = { points, ts };
    liveYaw += (yaw - liveYaw) * YAW_FOLLOW;
    // Start a swap when the model is free; frames keep flowing meanwhile, drawn with the last result.
    const latent = currentLatent(liveYaw);
    if (strength > 0.001 && !busy && latent && swapper) {
      busy = true;
      const local = snapshot(liveRegion, frame, points);
      swapJob(local, latent, w, h).catch((err) => post({ type: 'log', message: `swap failed: ${err}` })).finally(() => { busy = false; });
    }
  } else {
    smooth.reset(); lastLive = null;
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
      const embedding = m.views ? null : m.embedding ?? (m.image && m.landmarks ? await identityFromImage(m.image, m.landmarks) : null);
      m.image?.close();
      const views = m.views ?? (embedding ? [{ embedding, yaw: 0 }] : null);
      if (!views?.length) throw new Error('No face found in that picture.');
      // Mid-morph, the face on screen is a blend: carry it on as a single-angle identity.
      latentFrom = !latentTo ? null : latentFrom ? [{ latent: currentLatent(liveYaw)!, yaw: 0 }] : latentTo;
      latentTo = poseLatents(views);
      morphStart = performance.now(); morphMs = latentFrom ? m.morphMs : 0;
      if (strengthTo === 0) { strengthFrom = currentStrength(); strengthTo = 1; strengthStart = performance.now(); strengthMs = m.morphMs || 300; }
      // The most frontal photo's identity, to store.
      const front = views.reduce((a, b) => (Math.abs(b.yaw) < Math.abs(a.yaw) ? b : a)).embedding.slice();
      post({ type: 'identity', id: m.id, embedding: front }, [front.buffer]);
    } else if (m.type === 'embed') {
      const embeddings: Float32Array[] = [];
      for (const lm of m.faces) embeddings.push(await identityFromImage(m.image, lm));
      m.image.close();
      post({ type: 'embeddings', id: m.id, embeddings }, embeddings.map((e) => e.buffer));
    } else if (m.type === 'still') {
      const w = m.frame.width, h = m.frame.height;
      const faces = [];
      for (const f of m.faces) {
        const target = f.identity ? latentAt(poseLatents(f.identity), f.yaw) : currentLatent(f.yaw);
        if (!target) continue;
        const local = snapshot(stillRegion, m.frame, f.points);
        const { image, uv } = await swapAt(stillRegion, local, target, f.key, m.mask && { mask: m.mask, w, h });
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
      const canvas = render(m.frame, m.landmarks, m.yaw, m.ts);
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
