import { createFaceSwap, ISSUE_TEXT, KNOWN_SHA256, probe, type Backend, type FaceSwap, type FoundFace } from '../src/index.ts';

// Models: local files on a dev machine (see models/README.md), public mirrors elsewhere (the hosted demo).
// ?models=https://…/ points at your own copies. Every file is checked against its SHA-256 either way.
const param = new URLSearchParams(location.search).get('models');
const local = param ?? (['localhost', '127.0.0.1'].includes(location.hostname) ? new URL('models/', location.href).href : null);
const HF = 'https://huggingface.co/';
const models = {
  landmarker: { url: local ? local + 'face_landmarker.task' : 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', sha256: KNOWN_SHA256.landmarker },
  swapperFp16: { url: local ? local + 'inswapper_128_fp16.onnx' : HF + 'facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx', sha256: KNOWN_SHA256.swapperFp16 },
  swapperFp32: { url: local ? local + 'inswapper_128.onnx' : HF + 'facefusion/models-3.0.0/resolve/main/inswapper_128.onnx', sha256: KNOWN_SHA256.swapperFp32 },
  arcface: { url: local ? local + 'w600k_r50.onnx' : HF + 'public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx', sha256: KNOWN_SHA256.arcface },
  segmenter: { url: local ? local + 'selfie_multiclass_256x256.tflite' : 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite', sha256: KNOWN_SHA256.segmenter },
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const output = $<HTMLVideoElement>('output'), original = $<HTMLVideoElement>('original');
const status = $('status'), log = $('log'), progress = $<HTMLProgressElement>('progress');
const backendSel = $<HTMLSelectElement>('backend'), morph = $<HTMLInputElement>('morph'), strength = $<HTMLInputElement>('strength');

type Identity = { image: string; face?: FoundFace } | null;
let swap: FaceSwap | null = null;
let input: MediaStreamTrack | null = null;
let swapped: MediaStreamTrack | null = null;
let identity: Identity = null;
let stats = { fps: 0, inferMs: 0, backend: '', latencyMs: 0 };
const logs: string[] = [];

// Only offer what this browser can run.
void probe().then((caps) => {
  const off: Record<string, string | null> = {
    webgpu: caps.webgpu.available ? null : 'not available in this browser',
    webnn: caps.webnn ? null : 'off in this browser',
  };
  for (const o of backendSel.options) {
    const why = off[o.value];
    if (why) { o.disabled = true; o.textContent = `${o.dataset.label}: ${why}`; }
  }
  if (!caps.webnn) backendSel.title = 'Neural engine: in Chrome or Edge, turn on chrome://flags/#web-machine-learning-neural-network and relaunch.';
});

async function start(track: MediaStreamTrack) {
  stopRecording();
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
      onStats: (s) => { stats = s; status.textContent = `${s.backend.toUpperCase()}: ${Math.round(s.fps)} fps, ${Math.round(1000 / Math.max(s.inferMs, 1))} swaps/s (${Math.round(s.inferMs)} ms), ${s.latencyMs} ms latency`; },
      onLog: (m) => { logs.push(m); console.info('[faceswap]', m); },
    });
  } catch (err) {
    status.textContent = 'Couldn’t start';
    log.textContent = (err as Error).message;
    return;
  } finally { progress.hidden = true; }
  status.textContent = `Running on ${swap.backend.toUpperCase()}`;
  swapped = swap.process(track);
  output.srcObject = new MediaStream([swapped]);
  $<HTMLButtonElement>('rec').disabled = false;
  if (identity) await choose(identity, 0);
}

async function choose(next: Identity, ms = Number(morph.value)) {
  identity = next;
  if (!swap) return;
  log.textContent = '';
  try {
    if (!next) { swap.setStrength(0, 400); return; }
    await swap.setIdentity(next, { morphMs: ms });
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

const faces = document.querySelector<HTMLElement>('.faces')!;
const select = (b: HTMLButtonElement) => faces.querySelectorAll('.face').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
for (const b of faces.querySelectorAll<HTMLButtonElement>('.face')) {
  b.onclick = () => { select(b); void choose(b.dataset.face ? { image: b.dataset.face } : null); };
}

// Your own photo: find the faces, let the person pick one and see what might make it less accurate.
const dialog = $<HTMLDialogElement>('photo');
$<HTMLInputElement>('upload').onchange = async (e) => {
  const el = e.target as HTMLInputElement, file = el.files?.[0];
  el.value = '';
  if (!file) return;
  if (!swap) { log.textContent = 'Start your camera or the sample video first, then add a photo.'; return; }
  const url = URL.createObjectURL(file);
  log.textContent = 'Looking for faces…';
  let found: FoundFace[];
  try { found = await swap.findFaces(url); } catch (err) { log.textContent = (err as Error).message; return; }
  log.textContent = '';
  showPhoto(url, found);
};

function showPhoto(url: string, found: FoundFace[]) {
  const img = $<HTMLImageElement>('photoImg'), boxes = $('photoBoxes'), issues = $('photoIssues'), use = $<HTMLButtonElement>('photoUse');
  const preview = $<HTMLCanvasElement>('photoFace');
  let picked = found[0];
  img.src = url;
  boxes.replaceChildren();
  $('photoTitle').textContent = found.length ? 'Use this face?' : 'No face found';
  $('photoPick').hidden = found.length < 2;
  $('photoChosen').hidden = !found.length;
  use.hidden = !found.length;
  const show = (f: FoundFace) => {
    picked = f;
    preview.getContext('2d')!.drawImage(f.preview, 0, 0, 160, 160);
    issues.replaceChildren(...(f.issues.length
      ? f.issues.map((i) => Object.assign(document.createElement('li'), { textContent: ISSUE_TEXT[i] }))
      : [Object.assign(document.createElement('li'), { className: 'ok', textContent: 'Good photo: clear, sharp and facing the camera.' })]));
    boxes.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-pressed', String(found[i] === f)));
  };
  const place = () => {
    // The image is letterboxed (object-fit: contain): position the face boxes on what's actually shown.
    const k = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    const dx = (img.clientWidth - img.naturalWidth * k) / 2, dy = (img.clientHeight - img.naturalHeight * k) / 2;
    boxes.replaceChildren(...found.map((f, i) => {
      const b = Object.assign(document.createElement('button'), { className: 'photo-box', onclick: () => show(f) });
      b.setAttribute('aria-label', `Face ${i + 1}`);
      Object.assign(b.style, { left: `${dx + f.box.x * k}px`, top: `${dy + f.box.y * k}px`, width: `${f.box.width * k}px`, height: `${f.box.height * k}px` });
      return b;
    }));
    if (found.length) show(picked);
  };
  img.onload = place;
  if (!found.length) issues.replaceChildren();
  $('photoCancel').onclick = () => dialog.close();
  use.onclick = () => {
    dialog.close();
    const chip = Object.assign(document.createElement('button'), { className: 'face' });
    const thumb = document.createElement('canvas');
    thumb.width = thumb.height = 64;
    thumb.getContext('2d')!.drawImage(picked.preview, 0, 0, 64, 64);
    thumb.style.cssText = 'width:64px;height:64px;border-radius:8px';
    chip.append(thumb, 'Your photo');
    const id: Identity = { image: url, face: picked };
    chip.onclick = () => { select(chip); void choose(id); };
    faces.append(chip);
    select(chip);
    void choose(id);
  };
  dialog.showModal();
}

morph.oninput = () => { $('morphOut').textContent = `${(Number(morph.value) / 1000).toFixed(1)} s`; };
strength.oninput = () => {
  $('strengthOut').textContent = `${strength.value}%`;
  if (identity) swap?.setStrength(Number(strength.value) / 100, 150);
};

// Recording, in the browser: MP4 where supported (Chrome, Safari), WebM otherwise.
let recorder: MediaRecorder | null = null;
const rec = $<HTMLButtonElement>('rec'), download = $<HTMLAnchorElement>('download');
function stopRecording() { if (recorder?.state === 'recording') recorder.stop(); }
rec.onclick = () => {
  if (recorder?.state === 'recording') { stopRecording(); return; }
  if (!swapped) return;
  const type = ['video/mp4;codecs=avc1.640028', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find((t) => MediaRecorder.isTypeSupported(t)) ?? '';
  const chunks: Blob[] = [];
  const started = performance.now();
  recorder = new MediaRecorder(new MediaStream([swapped]), { mimeType: type, videoBitsPerSecond: 6_000_000 });
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    clearInterval(timer);
    rec.textContent = 'Record'; rec.setAttribute('aria-pressed', 'false');
    const blob = new Blob(chunks, { type: recorder!.mimeType });
    const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
    if (download.href) URL.revokeObjectURL(download.href);
    download.href = URL.createObjectURL(blob);
    download.download = `heisenbert-faceswap-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.${ext}`;
    download.textContent = `Download recording (${Math.round((performance.now() - started) / 1000)} s, ${(blob.size / 1e6).toFixed(1)} MB)`;
    download.hidden = false;
  };
  recorder.start(1000);
  rec.setAttribute('aria-pressed', 'true');
  const timer = setInterval(() => { rec.textContent = `Stop recording (${Math.round((performance.now() - started) / 1000)} s)`; }, 500);
  rec.textContent = 'Stop recording (0 s)';
};

// For automated checks.
Object.assign(window, { faceswapDemo: { get swap() { return swap; }, get stats() { return stats; }, logs, output, choose, start, sampleTrack, createFaceSwap, models, showPhoto } });
