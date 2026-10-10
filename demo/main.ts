import { createFaceSwap, ISSUE_TEXT, KNOWN_SHA256, probe, type Backend, type FaceSwap, type FoundFace, type VideoScan } from '../src/index.ts';

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
  // Macs gain the most (Core ML): show how to switch it on.
  $('webnnTip').hidden = caps.webnn || !/Mac/i.test(navigator.platform || navigator.userAgent);
});

for (const b of document.querySelectorAll<HTMLButtonElement>('[data-copy]')) {
  b.onclick = async () => {
    try { await navigator.clipboard.writeText(b.dataset.copy!); b.textContent = 'Copied'; } catch { b.textContent = 'Select and copy'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1500);
  };
}

/** The engine, created on first use (camera, sample, photo or video); the backend menu recreates it. */
async function ensureEngine() {
  if (swap) return swap;
  status.textContent = 'Loading models…';
  progress.hidden = false;
  try {
    swap = await createFaceSwap({
      models,
      backend: backendSel.value as Backend | 'auto',
      onProgress: (p) => {
        progress.max = p.total || 1; progress.value = p.loaded;
        status.textContent = p.loaded < p.total || !swap ? `Loading ${p.model}… ${Math.round(p.loaded / 1e6)} MB` : `Ready on ${swap.backend.toUpperCase()}`;
      },
      onStats: (s) => { stats = s; status.textContent = `${s.backend.toUpperCase()}: ${Math.round(s.fps)} fps, ${Math.round(1000 / Math.max(s.inferMs, 1))} swaps/s (${Math.round(s.inferMs)} ms), ${s.latencyMs} ms latency`; },
      onLog: (m) => { logs.push(m); console.info('[faceswap]', m); },
    });
  } catch (err) {
    status.textContent = 'Couldn’t start';
    log.textContent = (err as Error).message;
    return null;
  } finally { progress.hidden = true; }
  status.textContent = `Ready on ${swap.backend.toUpperCase()}`;
  if (identity) await choose(identity, 0);
  return swap;
}

async function start(track: MediaStreamTrack) {
  stopRecording();
  input?.stop(); swapped?.stop();
  input = track;
  original.srcObject = new MediaStream([track]); original.hidden = false;
  output.srcObject = new MediaStream([track]);
  if (!(await ensureEngine())) return;
  status.textContent = `Running on ${swap!.backend.toUpperCase()}`;
  swapped = swap!.process(track);
  output.srcObject = new MediaStream([swapped]);
  $<HTMLButtonElement>('rec').disabled = false;
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
backendSel.onchange = async () => {
  swap?.dispose(); swap = null; swapped?.stop();
  if (input) await start(input.clone()); else await ensureEngine();
};

const faces = document.querySelector<HTMLElement>('.faces')!;

// Faces that people in a video can become: the presets, plus every photo added.
type Target = { key: string; label: string; image: string; face?: FoundFace; thumb: CanvasImageSource };
const targets: Target[] = [...faces.querySelectorAll<HTMLButtonElement>('.face[data-face]')]
  .filter((b) => b.dataset.face)
  .map((b) => ({ key: b.dataset.face!, label: b.textContent!.trim(), image: b.dataset.face!, thumb: b.querySelector('img')! }));
let identityKey: string | null = null;
const select = (b: HTMLButtonElement) => faces.querySelectorAll('.face').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
for (const b of faces.querySelectorAll<HTMLButtonElement>('.face')) {
  b.onclick = () => { select(b); identityKey = b.dataset.face || null; void choose(b.dataset.face ? { image: b.dataset.face } : null); };
}

// Your own photo: find the faces, let the person pick one and see what might make it less accurate.
const dialog = $<HTMLDialogElement>('photo');
$<HTMLInputElement>('upload').onchange = async (e) => {
  const el = e.target as HTMLInputElement, file = el.files?.[0];
  el.value = '';
  if (!file) return;
  const url = URL.createObjectURL(file);
  log.textContent = 'Looking for faces…';
  let found: FoundFace[];
  try { found = await (await ensureEngine())!.findFaces(url); } catch (err) { log.textContent = (err as Error).message; return; }
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
    const key = `photo-${targets.length}`;
    targets.push({ key, label: `Your photo ${targets.filter((t) => t.key.startsWith('photo-')).length + 1}`, image: url, face: picked, thumb });
    chip.onclick = () => { select(chip); identityKey = key; void choose(id); };
    faces.append(chip);
    select(chip);
    identityKey = key;
    void choose(id);
    renderMapping();
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

// Video files: pick up to 60 s, find the people in it, choose who each becomes, swap every frame, download.
const MAX_SECONDS = 60;
let video: { file: File; duration: number; scan: VideoScan | null; range: [number, number] } | null = null;
const mapping = new Map<number, string>(); // person id → target key ('' = keep as is)
const identities = new Map<string, Float32Array>(); // target key → identity, computed once
let videoAbort: AbortController | null = null;
const vStatus = $('videoStatus'), vProgress = $<HTMLProgressElement>('videoProgress'), vGo = $<HTMLButtonElement>('videoGo');
const vResult = $<HTMLVideoElement>('videoResult'), vDownload = $<HTMLAnchorElement>('videoDownload'), vSource = $<HTMLVideoElement>('videoSource');
const tStart = $<HTMLInputElement>('trimStart'), tEnd = $<HTMLInputElement>('trimEnd');
const fmtTime = (s: number) => { const m = Math.floor(Math.round(s) / 60), r = Math.round(s) % 60; return m ? `${m} min${r ? ` ${r} s` : ''}` : `${r} s`; };
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

$<HTMLInputElement>('videoFile').onchange = async (e) => {
  const el = e.target as HTMLInputElement, file = el.files?.[0];
  el.value = '';
  if (!file) return;
  if (vSource.src) URL.revokeObjectURL(vSource.src);
  vSource.src = URL.createObjectURL(file);
  vResult.hidden = true; vDownload.hidden = true; $('mapping').hidden = true; vGo.disabled = true;
  const engine = await ensureEngine();
  if (!engine) return;
  let info;
  try { info = await engine.videoInfo(file); } catch (err) { vStatus.textContent = (err as Error).message; $('trimmer').hidden = true; return; }
  const duration = info.duration;
  video = { file, duration, scan: null, range: [0, Math.min(duration, MAX_SECONDS)] };
  $('videoInfo').textContent = `${file.name}: ${info.width}×${info.height}, ${fmtTime(duration)}, ${Math.round(info.fps)} fps`;
  for (const t of [tStart, tEnd]) t.max = String(duration);
  tStart.value = '0'; tEnd.value = String(video.range[1]);
  $('trimmer').hidden = false;
  showRange();
  vStatus.textContent = duration > MAX_SECONDS
    ? `Videos can be up to ${MAX_SECONDS} seconds: choose the part to use, then find the people in it.`
    : '';
  if (duration <= MAX_SECONDS) await findPeople(); // short enough: no trimming needed to start
};

// Start and end handles: at most 60 s apart, at least 1 s; dragging one pushes the other.
function onTrim(moved: 'start' | 'end') {
  if (!video) return;
  let s = Number(tStart.value), e = Number(tEnd.value);
  if (moved === 'start') { if (e - s > MAX_SECONDS) e = s + MAX_SECONDS; if (e - s < 1) e = Math.min(video.duration, s + 1); if (e - s < 1) s = e - 1; }
  else { if (e - s > MAX_SECONDS) s = e - MAX_SECONDS; if (e - s < 1) s = Math.max(0, e - 1); if (e - s < 1) e = s + 1; }
  tStart.value = String(s); tEnd.value = String(e);
  video.range = [s, e];
  vSource.pause(); vSource.currentTime = moved === 'start' ? s : e;
  showRange();
  if (video.scan) { video.scan = null; $('mapping').hidden = true; vGo.disabled = true; vStatus.textContent = 'The selection changed: find the people in it again.'; }
}
tStart.oninput = () => onTrim('start');
tEnd.oninput = () => onTrim('end');
function showRange() {
  const [s, e] = video!.range;
  $('trimStartOut').textContent = clock(s); $('trimEndOut').textContent = clock(e);
  $('trimInfo').textContent = `Selected ${clock(s)} to ${clock(e)} (${fmtTime(e - s)} of ${fmtTime(video!.duration)})`;
}
$('trimPlay').onclick = () => {
  if (!video) return;
  const [s, e] = video.range;
  vSource.currentTime = s; void vSource.play();
  vSource.ontimeupdate = () => { if (vSource.currentTime >= e) { vSource.pause(); vSource.ontimeupdate = null; } };
};
$('findPeople').onclick = () => void findPeople();

async function findPeople() {
  if (!video) return;
  const engine = await ensureEngine();
  if (!engine) return;
  const [start, end] = video.range;
  $('mapping').hidden = true; vGo.disabled = true;
  vStatus.textContent = 'Looking for people in this part…';
  try {
    const scan = await engine.scanVideo(video.file, { start, end });
    if (!video || video.range[0] !== start || video.range[1] !== end) return; // the selection moved meanwhile
    video.scan = scan;
    mapping.clear();
    if (!scan.people.length) { vStatus.textContent = 'No faces found in this part.'; return; }
    for (const p of scan.people) mapping.set(p.id, identityKey ?? targets[0]?.key ?? '');
    renderMapping();
    $('mapping').hidden = false;
    vStatus.textContent = `${scan.people.length === 1 ? 'One person' : `${scan.people.length} people`} found.`;
  } catch (err) { vStatus.textContent = (err as Error).message; }
}

/** One row per person: their face, then a menu of who they become. */
function renderMapping() {
  const people = video?.scan?.people ?? [];
  $('mapList').replaceChildren(...people.map((p) => {
    const li = document.createElement('li');
    const face = document.createElement('canvas');
    face.width = face.height = 112;
    face.getContext('2d')!.drawImage(p.preview, 0, 0, 112, 112);
    const target = document.createElement('canvas');
    target.width = target.height = 112;
    const sel = document.createElement('select');
    sel.setAttribute('aria-label', `Person ${p.id + 1} becomes`);
    sel.append(new Option('Keep as is', ''), ...targets.map((t) => new Option(t.label, t.key)));
    sel.value = mapping.get(p.id) ?? '';
    const draw = () => {
      const ctx = target.getContext('2d')!;
      ctx.clearRect(0, 0, 112, 112);
      const t = targets.find((x) => x.key === sel.value);
      if (t) ctx.drawImage(t.thumb, 0, 0, 112, 112);
      target.style.visibility = t ? 'visible' : 'hidden';
    };
    sel.onchange = () => { mapping.set(p.id, sel.value); draw(); updateGo(); };
    draw();
    const arrow = Object.assign(document.createElement('span'), { className: 'arrow', textContent: 'becomes' });
    li.append(face, `Person ${p.id + 1}`, arrow, sel, target);
    return li;
  }));
  updateGo();
}
function updateGo() { vGo.disabled = !video?.scan || ![...mapping.values()].some(Boolean); }

vGo.onclick = async () => {
  if (!video?.scan || !swap) return;
  videoAbort = new AbortController();
  vGo.disabled = true; $('videoCancel').hidden = false; vProgress.hidden = false; vProgress.value = 0;
  vResult.hidden = true; vDownload.hidden = true;
  vStatus.textContent = 'Preparing the faces…';
  try {
    const map: Record<number, Float32Array | null> = {};
    for (const [person, key] of mapping) {
      const t = targets.find((x) => x.key === key);
      if (!t) { map[person] = null; continue; }
      if (!identities.has(key)) identities.set(key, await swap.identityOf({ image: t.image, face: t.face }));
      map[person] = identities.get(key)!;
    }
    const [start, end] = video.range;
    const blob = await swap.swapVideo(video.file, {
      start, end,
      people: video.scan.people,
      map,
      signal: videoAbort.signal,
      onProgress: ({ progress: p, etaSeconds }) => {
        vProgress.value = p;
        vStatus.textContent = `${Math.round(p * 100)}%${etaSeconds !== null ? `, about ${fmtTime(etaSeconds)} left` : ''}`;
      },
    });
    if (vDownload.href) URL.revokeObjectURL(vDownload.href);
    const url = URL.createObjectURL(blob);
    vResult.src = url; vResult.hidden = false;
    vDownload.href = url;
    vDownload.download = video.file.name.replace(/\.[^.]+$/, '') + '-faceswap.mp4';
    vDownload.textContent = `Download the video (${(blob.size / 1e6).toFixed(1)} MB)`;
    vDownload.hidden = false;
    vStatus.textContent = 'Done.';
  } catch (err) {
    vStatus.textContent = videoAbort.signal.aborted ? 'Canceled.' : `Couldn’t finish: ${(err as Error).message}`;
  } finally {
    updateGo(); $('videoCancel').hidden = true; vProgress.hidden = true; videoAbort = null;
  }
};
$('videoCancel').onclick = () => videoAbort?.abort();

// For automated checks.
Object.assign(window, { faceswapDemo: { get swap() { return swap; }, get stats() { return stats; }, logs, output, choose, start, sampleTrack, createFaceSwap, models, showPhoto } });
