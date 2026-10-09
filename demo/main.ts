import { createFaceSwap, KNOWN_SHA256, type Backend, type FaceSwap } from '../src/index.ts';

// Models are served next to the demo (see README: where to get them). ?models=https://… points elsewhere.
const base = new URLSearchParams(location.search).get('models') ?? new URL('models/', location.href).href;
const models = {
  landmarker: { url: base + 'face_landmarker.task', sha256: KNOWN_SHA256.landmarker },
  swapperFp16: { url: base + 'inswapper_128_fp16.onnx', sha256: KNOWN_SHA256.swapperFp16 },
  swapperFp32: { url: base + 'inswapper_128.onnx', sha256: KNOWN_SHA256.swapperFp32 },
  arcface: { url: base + 'w600k_r50.onnx', sha256: KNOWN_SHA256.arcface },
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const output = $<HTMLVideoElement>('output'), original = $<HTMLVideoElement>('original');
const status = $('status'), log = $('log'), progress = $<HTMLProgressElement>('progress');
const backendSel = $<HTMLSelectElement>('backend'), morph = $<HTMLInputElement>('morph'), strength = $<HTMLInputElement>('strength');

let swap: FaceSwap | null = null;
let input: MediaStreamTrack | null = null;
let swapped: MediaStreamTrack | null = null;
let face = '';
let stats = { fps: 0, inferMs: 0, backend: '' };

async function start(track: MediaStreamTrack) {
  input?.stop(); swapped?.stop(); swap?.dispose();
  input = track;
  original.srcObject = new MediaStream([track]); original.hidden = false;
  output.srcObject = new MediaStream([track]);
  status.textContent = 'Loading models…';
  progress.hidden = false;
  try {
    swap = await createFaceSwap({
      models,
      backend: backendSel.value as Backend | 'auto',
      onProgress: (p) => { progress.max = p.total || 1; progress.value = p.loaded; status.textContent = `Loading ${p.model}… ${Math.round(p.loaded / 1e6)} MB`; },
      onStats: (s) => { stats = s; status.textContent = `${s.backend.toUpperCase()}: ${Math.round(s.fps)} fps, ${Math.round(s.inferMs)} ms per swap`; },
      onLog: (m) => console.info('[faceswap]', m),
    });
  } catch (err) {
    status.textContent = 'Couldn’t start';
    log.textContent = (err as Error).message;
    return;
  } finally { progress.hidden = true; }
  status.textContent = `Running on ${swap.backend.toUpperCase()}`;
  swapped = swap.process(track);
  output.srcObject = new MediaStream([swapped]);
  if (face) await choose(face, 0);
}

async function choose(src: string, ms = Number(morph.value)) {
  face = src;
  if (!swap) return;
  log.textContent = '';
  try {
    if (!src) { swap.setStrength(0, 400); return; }
    await swap.setIdentity({ image: src }, { morphMs: ms });
    swap.setStrength(Number(strength.value) / 100, 400);
  } catch (err) { log.textContent = (err as Error).message; }
}

// A short looping clip, so the demo (and the tests) work without a camera. Video keeps playing in background tabs.
async function sampleTrack() {
  const v = document.createElement('video');
  v.src = 'sample.mp4'; v.muted = true; v.loop = true; v.playsInline = true;
  await v.play().catch(() => {});
  return (v as HTMLVideoElement & { captureStream(): MediaStream }).captureStream().getVideoTracks()[0];
}

$('camera').onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    await start(s.getVideoTracks()[0]);
  } catch (err) { log.textContent = `Camera unavailable: ${(err as Error).message}`; }
};
$('sample').onclick = async () => {
  const track = await sampleTrack();
  if (!track) { log.textContent = 'This browser can’t capture video from a clip.'; return; }
  await start(track);
};
backendSel.onchange = () => input && start(input.clone());

for (const b of document.querySelectorAll<HTMLButtonElement>('.face')) {
  b.onclick = () => {
    document.querySelectorAll('.face').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    void choose(b.dataset.face ?? '');
  };
}
$<HTMLInputElement>('upload').onchange = (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  document.querySelectorAll('.face').forEach((x) => x.setAttribute('aria-pressed', 'false'));
  void choose(URL.createObjectURL(file));
};
morph.oninput = () => { $('morphOut').textContent = `${(Number(morph.value) / 1000).toFixed(1)} s`; };
strength.oninput = () => {
  $('strengthOut').textContent = `${strength.value}%`;
  if (face) swap?.setStrength(Number(strength.value) / 100, 150);
};

// For automated checks.
Object.assign(window, { faceswapDemo: { get swap() { return swap; }, get stats() { return stats; }, output, choose, start, sampleTrack } });
