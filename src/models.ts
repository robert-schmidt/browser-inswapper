// Model files: fetched once with progress, kept in the browser's Cache Storage, checked against a SHA-256
// before use. No weights ship with this package; the app says where they live (see README: licences).

export type ModelSpec = { url: string; sha256?: string; bytes?: number };
export type ModelSet = {
  /** MediaPipe Face Landmarker (.task). Apache-2.0; defaults to Google's hosted copy. */
  landmarker?: ModelSpec;
  /** inswapper_128 in fp16 (smaller, for GPUs with shader-f16) and/or fp32 (CPU and older GPUs). */
  swapperFp16?: ModelSpec;
  swapperFp32?: ModelSpec;
  /** ArcFace w600k_r50: only needed to swap to a face from a photo (identities can be passed as embeddings). */
  arcface?: ModelSpec;
  /** MediaPipe multiclass selfie segmenter (.tflite): keeps hands and objects in front of the face. Apache-2.0; defaults to Google's hosted copy. */
  segmenter?: ModelSpec;
};

/** Fingerprints of the public builds this package was developed against. */
export const KNOWN_SHA256 = {
  landmarker: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
  swapperFp16: 'c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420',
  swapperFp32: 'a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7',
  arcface: '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
  segmenter: 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0',
} as const;

export const DEFAULT_LANDMARKER: ModelSpec = {
  url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  sha256: KNOWN_SHA256.landmarker,
  bytes: 3_758_596,
};

export const DEFAULT_SEGMENTER: ModelSpec = {
  url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite',
  sha256: KNOWN_SHA256.segmenter,
  bytes: 16_371_837,
};

export type Progress = { model: string; loaded: number; total: number };
// ponytail: pre-rename name kept so returning visitors don't download the models again.
const CACHE = 'heisenbert-faceswap-models-v1';

async function sha256(buf: ArrayBuffer) {
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Returns the model bytes: from the cache when present (and matching), otherwise downloaded and cached. */
export async function loadModel(name: string, spec: ModelSpec, onProgress?: (p: Progress) => void, signal?: AbortSignal): Promise<ArrayBuffer> {
  const cache = typeof caches !== 'undefined' ? await caches.open(CACHE).catch(() => null) : null;
  const key = spec.sha256 ? `${spec.url}#${spec.sha256}` : spec.url;
  const hit = await cache?.match(key);
  if (hit) {
    const buf = await hit.arrayBuffer();
    onProgress?.({ model: name, loaded: buf.byteLength, total: buf.byteLength });
    return buf;
  }
  const res = await fetch(spec.url, { signal, credentials: 'omit' });
  if (!res.ok || !res.body) throw new Error(`Couldn’t download ${name} (${res.status}).`);
  const total = Number(res.headers.get('content-length')) || spec.bytes || 0;
  // Into one buffer as it arrives, not chunks plus a joined copy plus a copy for the cache: phones kill a tab
  // that holds three copies of a 280 MB model at once (it looks like the page reloading by itself).
  const reader = res.body.getReader();
  let bytes = new Uint8Array(total);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.byteLength > bytes.byteLength) { // no or wrong length header: grow
      const grown = new Uint8Array(Math.max(loaded + value.byteLength, bytes.byteLength * 2));
      grown.set(bytes.subarray(0, loaded)); bytes = grown;
    }
    bytes.set(value, loaded);
    loaded += value.byteLength;
    onProgress?.({ model: name, loaded, total: Math.max(total, loaded) });
  }
  if (loaded < bytes.byteLength) bytes = bytes.slice(0, loaded);
  if (spec.sha256 && (await sha256(bytes.buffer)) !== spec.sha256) throw new Error(`${name} didn’t match its fingerprint; not used.`);
  // Awaited before returning: the caller hands the buffer to a worker, which empties it here.
  await cache?.put(key, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
  return bytes.buffer;
}

/** Removes cached models (e.g. a "free up space" button). */
export async function clearModelCache() {
  if (typeof caches !== 'undefined') await caches.delete(CACHE);
}
