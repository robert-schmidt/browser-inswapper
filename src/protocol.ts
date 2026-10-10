// Messages between the page (index.ts) and the swap worker (worker.ts).
import type { Backend, Capabilities } from './backend.ts';
import type { Mesh } from './mesh.ts';

export type ToWorker =
  | { type: 'init'; swapper: ArrayBuffer; arcface?: ArrayBuffer; backend: Backend | 'auto'; caps: Capabilities; wasmPaths?: string; mesh: Mesh; masks?: MessagePort }
  | { type: 'arcface'; bytes: ArrayBuffer }
  | { type: 'identity'; id: number; embedding?: Float32Array; image?: ImageBitmap; landmarks?: Float32Array; morphMs: number }
  | { type: 'strength'; value: number; ms: number }
  | { type: 'frame'; id: number; frame: VideoFrame | ImageBitmap; landmarks: Float32Array | null; ts: number }
  // Video files: identity embeddings for faces in a frame, and a frame with every listed face freshly swapped.
  | { type: 'embed'; id: number; image: ImageBitmap; faces: Float32Array[] }
  | { type: 'still'; id: number; frame: ImageBitmap; faces: { key: string; points: Float32Array }[]; mask: FaceMask | null };

/** Where the swapped face may show (0–255: 0 where a hand or object is in front), low resolution, whole frame. */
export type FaceMask = { data: Uint8Array; width: number; height: number };

export type ToMain =
  | { type: 'ready'; backend: Backend; inferMs: number }
  | { type: 'identity'; id: number; embedding: Float32Array }
  | { type: 'embeddings'; id: number; embeddings: Float32Array[] }
  | { type: 'frame'; id: number; frame: VideoFrame | ImageBitmap }
  | { type: 'stats'; fps: number; inferMs: number; backend: Backend }
  | { type: 'log'; message: string }
  | { type: 'error'; id?: number; message: string };
