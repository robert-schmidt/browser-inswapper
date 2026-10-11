// Messages between the page (index.ts) and the swap worker (worker.ts).
import type { Backend, Capabilities } from './backend.ts';
import type { Mesh } from './mesh.ts';

export type ToWorker =
  | { type: 'init'; swapper: ArrayBuffer; arcface?: ArrayBuffer; backend: Backend | 'auto'; caps: Capabilities; wasmPaths?: string; mesh: Mesh; masks?: MessagePort }
  | { type: 'arcface'; bytes: ArrayBuffer }
  // `views`: one identity per photo angle (picked per frame by head turn); otherwise one embedding, or a photo to read it from.
  | { type: 'identity'; id: number; views?: IdentityView[]; embedding?: Float32Array; image?: ImageBitmap; landmarks?: Float32Array; morphMs: number }
  | { type: 'strength'; value: number; ms: number }
  | { type: 'frame'; id: number; frame: VideoFrame | ImageBitmap; landmarks: Float32Array | null; yaw: number; ts: number }
  // Video files: identity embeddings for faces in a frame, and a frame with every listed face freshly swapped.
  | { type: 'embed'; id: number; image: ImageBitmap; faces: Float32Array[] }
  // `identity`: that face's own target; without it, the current identity is used. `yaw`: its head turn (see yawOf).
  | { type: 'still'; id: number; frame: ImageBitmap; faces: { key: string; points: Float32Array; yaw: number; identity?: IdentityView[] }[]; mask: FaceMask | null };

/** A face's identity (ArcFace, 512 numbers) as seen at one head turn, in degrees (0 = facing the camera). */
export type IdentityView = { embedding: Float32Array; yaw: number };

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
