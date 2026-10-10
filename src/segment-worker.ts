/// <reference lib="webworker" />
// Occlusion: finds face skin in small copies of the frame, in its own worker so it never holds up the video.
// Masks go straight to the swap worker over a MessagePort; the swap worker always uses the latest one.

import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import type { FaceMask } from './protocol.ts';

type In =
  | { type: 'init'; model: ArrayBuffer; wasm: string; port?: MessagePort }
  // `reply`: send the mask back to the sender (video files need the mask of this exact frame).
  | { type: 'frame'; bitmap: ImageBitmap; ts: number; reply?: boolean };

let segmenter: ImageSegmenter | null = null;
let port: MessagePort | undefined;
// selfie_multiclass_256x256 classes: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other.
// The mask says where the swapped face may show: everywhere except what can be in front of a face (hands are
// body skin; mics and cups are "other"). Not "face skin only": eyes and lips aren't face skin, and the real
// ones would show through.
const IN_FRONT = [2, 4, 5];

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
      const masks = IN_FRONT.map((c) => r.confidenceMasks?.[c]?.getAsFloat32Array());
      if (masks.every(Boolean)) {
        const n = masks[0]!.length;
        const mask: FaceMask = { data: new Uint8Array(n), width: m.bitmap.width, height: m.bitmap.height };
        for (let i = 0; i < n; i++) mask.data[i] = Math.max(0, 1 - masks[0]![i] - masks[1]![i] - masks[2]![i]) * 255;
        if (m.reply) { self.postMessage({ type: 'mask', mask }, [mask.data.buffer]); return; }
        port?.postMessage(mask, [mask.data.buffer]);
      }
      if (m.reply) self.postMessage({ type: 'mask', mask: null });
    } finally { r.close(); }
  } catch { /* skip this one; the next frame tries again */ } finally {
    m.bitmap.close();
    self.postMessage({ type: 'done' });
  }
};
