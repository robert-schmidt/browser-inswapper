// Messages between the page (index.ts) and the swap worker (worker.ts).
import type { Backend, Capabilities } from './backend.ts';
import type { Mesh } from './mesh.ts';

export type ToWorker =
  | { type: 'init'; swapper: ArrayBuffer; arcface?: ArrayBuffer; backend: Backend | 'auto'; caps: Capabilities; wasmPaths?: string; mesh: Mesh }
  | { type: 'arcface'; bytes: ArrayBuffer }
  | { type: 'identity'; id: number; embedding?: Float32Array; image?: ImageBitmap; landmarks?: Float32Array; morphMs: number }
  | { type: 'strength'; value: number; ms: number }
  | { type: 'frame'; id: number; frame: VideoFrame | ImageBitmap; landmarks: Float32Array | null; ts: number };

export type ToMain =
  | { type: 'ready'; backend: Backend; inferMs: number }
  | { type: 'identity'; id: number; embedding: Float32Array }
  | { type: 'frame'; id: number; frame: VideoFrame | ImageBitmap }
  | { type: 'stats'; fps: number; inferMs: number; backend: Backend }
  | { type: 'log'; message: string }
  | { type: 'error'; id?: number; message: string };
