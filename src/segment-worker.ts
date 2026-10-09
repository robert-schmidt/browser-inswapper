/// <reference lib="webworker" />
// Occlusion: finds face skin in small copies of the frame, in its own worker so it never holds up the video.
// Masks go straight to the swap worker over a MessagePort; the swap worker always uses the latest one.

import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import type { FaceMask } from './protocol.ts';

type In =
  | { type: 'init'; model: ArrayBuffer; wasm: string; port: MessagePort }
  | { type: 'frame'; bitmap: ImageBitmap; ts: number };

let segmenter: ImageSegmenter | null = null;
let port: MessagePort;
const FACE_SKIN = 3; // selfie_multiclass_256x256: background, hair, body skin, face skin, clothes, other

self.onmessage = async (e: MessageEvent<In>) => {
  const m = e.data;
  if (m.type === 'init') {
    port = m.port;
    try {
      const vision = await FilesetResolver.forVisionTasks(m.wasm, true);
      const opts = { runningMode: 'VIDEO' as const, outputConfidenceMasks: true, outputCategoryMask: false, canvas: new OffscreenCanvas(1, 1) };
      segmenter = await ImageSegmenter.createFromOptions(vision, { ...opts, baseOptions: { modelAssetBuffer: new Uint8Array(m.model), delegate: 'GPU' } })
        .catch(() => ImageSegmenter.createFromOptions(vision, { ...opts, canvas: undefined, baseOptions: { modelAssetBuffer: new Uint8Array(m.model), delegate: 'CPU' } }));
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  try {
    if (!segmenter) return;
    const r = segmenter.segmentForVideo(m.bitmap, m.ts);
    try {
      const conf = r.confidenceMasks?.[FACE_SKIN]?.getAsFloat32Array();
      if (conf) {
        const mask: FaceMask = { data: new Uint8Array(conf.length), width: m.bitmap.width, height: m.bitmap.height };
        for (let i = 0; i < conf.length; i++) mask.data[i] = conf[i] * 255;
        port.postMessage(mask, [mask.data.buffer]);
      }
    } finally { r.close(); }
  } catch { /* skip this one; the next frame tries again */ } finally {
    m.bitmap.close();
    self.postMessage({ type: 'done' });
  }
};
