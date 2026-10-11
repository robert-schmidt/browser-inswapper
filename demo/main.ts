import { createFaceSwap, ISSUE_TEXT, KNOWN_SHA256, probe, type Backend, type FaceSwap, type FoundFace, type IdentityView, type VideoScan } from '../src/index.ts';

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
const backendSel = $<HTMLSelectElement>('backend'), videoBackend = $<HTMLSelectElement>('videoBackend'), strength = $<HTMLInputElement>('strength');

type Identity = { image: string; face?: FoundFace } | { views: IdentityView[] } | null;
let swap: FaceSwap | null = null;
let input: MediaStreamTrack | null = null;
let mic: MediaStreamTrack | null = null; // the camera's sound, recorded with the swapped video
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
  // The video section has its own copy of the menu: one engine, so both always show the same choice.
  videoBackend.replaceChildren(...[...backendSel.options].map((o) => o.cloneNode(true)));
  videoBackend.value = backendSel.value; videoBackend.title = backendSel.title;
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

async function choose(next: Identity, ms = 0) { // instant: the library can blend between faces (morphMs), the demo doesn't need to
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
    const video = { width: 1280, height: 720 };
    // With the microphone for recordings; without it if there's none or it's refused.
    const s = await navigator.mediaDevices.getUserMedia({ video, audio: true }).catch(() => navigator.mediaDevices.getUserMedia({ video }));
    mic?.stop(); mic = s.getAudioTracks()[0] ?? null;
    await start(s.getVideoTracks()[0]);
  } catch (err) { log.textContent = `Camera unavailable: ${(err as Error).message}`; }
};
$('sample').onclick = async () => {
  mic?.stop(); mic = null; // the sample has no sound
  const track = await sampleTrack();
  if (!track) { log.textContent = 'This browser can’t capture video from a clip.'; return; }
  await start(track);
};
videoBackend.onchange = () => { backendSel.value = videoBackend.value; void backendSel.onchange!(new Event('change')); };
backendSel.onchange = async () => {
  videoBackend.value = backendSel.value;
  swap?.dispose(); swap = null; swapped?.stop();
  if (input) await start(input.clone()); else await ensureEngine();
};

const faces = document.querySelector<HTMLElement>('.faces')!;

// Faces that people in a video can become: the presets, plus every photo added.
// `views`: the front photo plus side photos of the same person, once any are added (see addSides).
type Target = { key: string; label: string; image: string; face?: FoundFace; thumb: CanvasImageSource; views?: IdentityView[] };
const targets: Target[] = [...faces.querySelectorAll<HTMLButtonElement>('.face[data-face]')]
  .filter((b) => b.dataset.face)
  .map((b) => ({ key: b.dataset.face!, label: b.textContent!.trim(), image: b.dataset.face!, thumb: b.querySelector('img')! }));
let identityKey: string | null = null;
const select = (b: HTMLButtonElement) => faces.querySelectorAll('.face').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
for (const b of faces.querySelectorAll<HTMLButtonElement>('.face')) {
  b.onclick = () => { select(b); identityKey = b.dataset.face || null; showSides(); void choose(b.dataset.face ? { image: b.dataset.face } : null); };
}

// Your own photo: find the faces, let the person pick one and see what might make it less accurate.
const dialog = $<HTMLDialogElement>('photo');
$<HTMLInputElement>('upload').onchange = async (e) => {
  const el = e.target as HTMLInputElement, file = el.files?.[0];
  el.value = '';
  if (!file) return;
  log.textContent = 'Looking for faces…';
  let url: string, found: FoundFace[];
  try {
    url = await upright(file);
    found = await (await ensureEngine())!.findFaces(url);
  } catch (err) { log.textContent = (err as Error).message; return; }
  log.textContent = '';
  showPhoto(url, found);
};

/** The photo exactly as the page shows it (rotation applied), so detection, boxes and the swap all see the same pixels. */
async function upright(source: Blob | ImageBitmap, region?: { x: number; y: number; width: number; height: number }) {
  const bmp = source instanceof Blob ? await createImageBitmap(source, { imageOrientation: 'from-image' }) : source;
  const r = region ?? { x: 0, y: 0, width: bmp.width, height: bmp.height };
  const c = new OffscreenCanvas(Math.round(r.width), Math.round(r.height));
  c.getContext('2d')!.drawImage(bmp, r.x, r.y, r.width, r.height, 0, 0, c.width, c.height);
  if (source instanceof Blob) bmp.close();
  return URL.createObjectURL(await c.convertToBlob({ type: 'image/jpeg', quality: 0.95 }));
}

function showPhoto(url: string, found: FoundFace[]) {
  const img = $<HTMLImageElement>('photoImg'), boxes = $('photoBoxes'), issues = $('photoIssues'), use = $<HTMLButtonElement>('photoUse');
  const preview = $<HTMLCanvasElement>('photoFace'), wrap = img.parentElement!;
  // Each face remembers which picture its landmarks belong to: the photo, or a crop the person drew.
  const sources = new Map<FoundFace, string>(found.map((f) => [f, url]));
  let picked = found[0];
  const detected = found.length;
  img.src = url;
  boxes.replaceChildren();
  const titles = () => {
    $('photoTitle').textContent = found.length ? 'Use this face?' : 'No face found';
    $('photoPick').hidden = detected < 2;
    $('photoChosen').hidden = !found.length;
    use.hidden = !found.length;
  };
  titles();
  const show = (f: FoundFace) => {
    picked = f;
    preview.getContext('2d')!.drawImage(f.preview, 0, 0, 160, 160);
    issues.replaceChildren(...(f.issues.length
      ? f.issues.map((i) => Object.assign(document.createElement('li'), { textContent: ISSUE_TEXT[i] }))
      : [Object.assign(document.createElement('li'), { className: 'ok', textContent: 'Good photo: clear, sharp and facing the camera.' })]));
    boxes.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-pressed', String(found[i] === f)));
  };
  // The image is letterboxed (object-fit: contain): map between photo pixels and what's actually shown.
  const fit = () => {
    const k = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    return { k, dx: (img.clientWidth - img.naturalWidth * k) / 2, dy: (img.clientHeight - img.naturalHeight * k) / 2 };
  };
  const place = () => {
    const { k, dx, dy } = fit();
    boxes.replaceChildren(...found.map((f, i) => {
      const b = Object.assign(document.createElement('button'), { className: 'photo-box', onclick: () => show(f) });
      b.setAttribute('aria-label', `Face ${i + 1}`);
      Object.assign(b.style, { left: `${dx + f.box.x * k}px`, top: `${dy + f.box.y * k}px`, width: `${f.box.width * k}px`, height: `${f.box.height * k}px` });
      return b;
    }));
    if (found.length) show(picked);
  };
  img.onload = place;
  // Wrong face, or none found: drag a box around the right one and it's looked for in just that area.
  const hint = $('photoHint');
  hint.textContent = 'Wrong face, or none found? Drag a box around the face on the photo.';
  wrap.onpointerdown = (e) => {
    if (e.button !== 0) return;
    const r = wrap.getBoundingClientRect(), x0 = e.clientX - r.left, y0 = e.clientY - r.top;
    const draw = Object.assign(document.createElement('div'), { className: 'photo-box drawing' });
    let moved = false;
    wrap.onpointermove = (m) => {
      const x1 = m.clientX - r.left, y1 = m.clientY - r.top;
      if (!moved && Math.hypot(x1 - x0, y1 - y0) < 8) return; // a click selects a box instead
      if (!moved) { moved = true; boxes.append(draw); wrap.setPointerCapture(e.pointerId); }
      Object.assign(draw.style, { left: `${Math.min(x0, x1)}px`, top: `${Math.min(y0, y1)}px`, width: `${Math.abs(x1 - x0)}px`, height: `${Math.abs(y1 - y0)}px` });
    };
    wrap.onpointerup = async () => {
      wrap.onpointermove = wrap.onpointerup = null;
      if (!moved) return;
      const { k, dx, dy } = fit();
      const b = draw.getBoundingClientRect();
      // A square around the box with some margin: the detector wants the whole head and a bit around it.
      const cx = (b.left + b.width / 2 - r.left - dx) / k, cy = (b.top + b.height / 2 - r.top - dy) / k;
      const side = (1.4 * Math.max(b.width, b.height)) / k;
      const region = { x: cx - side / 2, y: cy - side / 2, width: side, height: side };
      draw.remove();
      if (side < 24) return;
      hint.textContent = 'Looking for a face there…';
      try {
        const crop = await upright(await createImageBitmap(img), region);
        const [f] = await swap!.findFaces(crop);
        if (!f) { URL.revokeObjectURL(crop); hint.textContent = 'No face in that area. Try a slightly bigger box around the whole head.'; return; }
        const s = side / Math.round(side); // the crop was rounded to whole pixels
        const face = { ...f, box: { x: region.x + f.box.x * s, y: region.y + f.box.y * s, width: f.box.width * s, height: f.box.height * s } };
        found.push(face);
        sources.set(face, crop);
        picked = face;
        titles();
        place();
        hint.textContent = 'Found it. Drag again to try another area.';
      } catch (err) { hint.textContent = (err as Error).message; }
    };
  };
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
    const image = sources.get(picked)!;
    const key = `photo-${targets.length}`;
    const target: Target = { key, label: `Your photo ${targets.filter((t) => t.key.startsWith('photo-')).length + 1}`, image, face: picked, thumb };
    targets.push(target);
    chip.onclick = () => { select(chip); identityKey = key; showSides(); void choose(identityOf(target)); };
    faces.append(chip);
    select(chip);
    identityKey = key;
    showSides();
    void choose(identityOf(target));
    renderMapping();
  };
  dialog.showModal();
}

// Side photos: the front photo alone gives every head turn the identity measured face-on, and the swap loses
// likeness when the head turns. With a photo turned each way, faces turned to the side use the matching one.
const sides = $('sides'), sideLog = $('sideLog');
const identityOf = (t: Target): Identity => (t.views && t.views.length > 1 ? { views: t.views } : { image: t.image, face: t.face });
const current = () => targets.find((t) => t.key === identityKey && t.key.startsWith('photo-'));
const cosine = (a: Float32Array, b: Float32Array) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y); };
const MIN_TURN = 10, MAX_TURN = 60; // degrees: less is just another front photo; more is a profile, where the identity model is unreliable
const SAME_PERSON = 0.3; // identity similarity below this: not the same person (or not a usable photo)
const thumbs = new Map<IdentityView, HTMLCanvasElement>();

function showSides() {
  const t = current();
  sides.hidden = !t;
  if (!t) return;
  $('sidesFor').textContent = t.label;
  const views = t.views ?? [];
  const front = views.find((v) => !thumbs.has(v));
  const turned = (v: IdentityView) => v.yaw - (front?.yaw ?? 0);
  $('sideViews').replaceChildren(...views.filter((v) => thumbs.has(v)).map((v) => {
    const fig = document.createElement('figure');
    fig.append(thumbs.get(v)!, Object.assign(document.createElement('figcaption'), { textContent: `${Math.round(Math.abs(turned(v)))}° ${turned(v) > 0 ? '→' : '←'}` }));
    return fig;
  }));
  const n = new Set(views.filter((v) => thumbs.has(v)).map((v) => Math.sign(turned(v)))).size;
  if (!sideLog.dataset.busy) sideLog.textContent = n === 2 ? 'Both sides added: faces turned to the side now use these photos.' : n === 1 ? 'One side added. Add one turned the other way too.' : '';
}

$<HTMLInputElement>('sideUpload').onchange = async (e) => {
  const el = e.target as HTMLInputElement, files = [...(el.files ?? [])];
  el.value = '';
  const t = current();
  if (!t || !files.length || !swap) return;
  const notes: string[] = [];
  sideLog.dataset.busy = '1';
  sideLog.textContent = 'Reading the photos…';
  try {
    t.views ??= [await swap.identityView({ image: t.image, face: t.face })];
    const front = t.views[0];
    for (const file of files) {
      const url = await upright(file);
      const [f] = await swap.findFaces(url);
      if (!f) { notes.push(`${file.name}: no face found.`); continue; }
      const view = await swap.identityView({ image: url, face: f });
      const turn = view.yaw - front.yaw;
      if (Math.abs(turn) < MIN_TURN) { notes.push(`${file.name}: faces the camera like the front photo. Turn the head a little more (about a quarter turn).`); continue; }
      if (Math.abs(turn) > MAX_TURN) { notes.push(`${file.name}: turned too far (${Math.round(Math.abs(turn))}°). Turn a little less, so both eyes still show.`); continue; }
      if (cosine(view.embedding, front.embedding) < SAME_PERSON) { notes.push(`${file.name}: doesn't look like the same person as the front photo.`); continue; }
      const thumb = document.createElement('canvas');
      thumb.width = thumb.height = 64;
      thumb.getContext('2d')!.drawImage(f.preview, 0, 0, 64, 64);
      thumbs.set(view, thumb);
      // One photo per side: a newer one replaces the older.
      t.views = [...t.views.filter((v) => v === front || Math.sign(v.yaw - front.yaw) !== Math.sign(turn)), view];
    }
  } catch (err) { notes.push((err as Error).message); }
  delete sideLog.dataset.busy;
  showSides();
  if (notes.length) sideLog.textContent = notes.join(' ');
  if (identityKey === t.key) void choose(identityOf(t), 0);
};

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
  const sound = mic?.readyState === 'live' ? mic : null;
  const types = sound
    ? ['video/mp4;codecs=avc1.640028,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm']
    : ['video/mp4;codecs=avc1.640028', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  const type = types.find((t) => MediaRecorder.isTypeSupported(t)) ?? '';
  const chunks: Blob[] = [];
  const started = performance.now();
  recorder = new MediaRecorder(new MediaStream(sound ? [swapped, sound] : [swapped]), { mimeType: type, videoBitsPerSecond: 6_000_000, audioBitsPerSecond: 128_000 });
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
  // One engine for both: pause the live preview so the video gets all of it (and it isn't slowed down).
  const resume = input?.readyState === 'live' && swapped ? input.clone() : null;
  stopRecording(); swapped?.stop(); swapped = null;
  if (resume) status.textContent = 'Paused while the video is processed';
  for (const el of [backendSel, videoBackend, rec, $<HTMLButtonElement>('camera'), $<HTMLButtonElement>('sample')]) el.disabled = true;
  vResult.hidden = true; vDownload.hidden = true;
  vStatus.textContent = 'Preparing the faces…';
  try {
    const map: Record<number, Float32Array | IdentityView[] | null> = {};
    for (const [person, key] of mapping) {
      const t = targets.find((x) => x.key === key);
      if (!t) { map[person] = null; continue; }
      if (t.views && t.views.length > 1) { map[person] = t.views; continue; }
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
        // Every frame has been read at 100%; the last ones are still being swapped and the file written.
        vStatus.textContent = p >= 0.999 ? 'Finishing the file…' : `${Math.round(p * 100)}%${etaSeconds !== null ? `, about ${fmtTime(etaSeconds)} left` : ''}`;
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
    for (const el of [backendSel, videoBackend, $<HTMLButtonElement>('camera'), $<HTMLButtonElement>('sample')]) el.disabled = false;
    if (resume) await start(resume);
  }
};
$('videoCancel').onclick = () => videoAbort?.abort();

// For automated checks.
Object.assign(window, { faceswapDemo: { get swap() { return swap; }, get stats() { return stats; }, logs, output, choose, start, sampleTrack, createFaceSwap, models, showPhoto } });
