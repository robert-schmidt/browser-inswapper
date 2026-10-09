// Picks how to run the models on this device: WebGPU (Metal on Apple Silicon, D3D12 or Vulkan elsewhere),
// WebNN (Core ML / DirectML / NPUs where the browser has it), or WebAssembly on the CPU. Probing is cheap;
// the final choice comes from a short benchmark in the worker and is remembered per device.

export type Backend = 'webgpu' | 'webnn' | 'wasm';
export type Capabilities = {
  webgpu: { available: boolean; vendor: string; architecture: string; f16: boolean };
  webnn: boolean;
  /** Multi-threaded WebAssembly needs a cross-origin isolated page (COOP + COEP headers). */
  wasmThreads: boolean;
  cores: number;
};

type GpuAdapterLike = { features: Set<string>; info?: { vendor?: string; architecture?: string } };

export async function probe(): Promise<Capabilities> {
  const nav = (typeof navigator !== 'undefined' ? navigator : {}) as Navigator & { gpu?: { requestAdapter(o?: object): Promise<GpuAdapterLike | null> }; ml?: unknown };
  let webgpu: Capabilities['webgpu'] = { available: false, vendor: '', architecture: '', f16: false };
  try {
    const adapter = await nav.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) webgpu = { available: true, vendor: adapter.info?.vendor ?? '', architecture: adapter.info?.architecture ?? '', f16: adapter.features.has('shader-f16') };
  } catch { /* no WebGPU */ }
  return {
    webgpu,
    webnn: !!nav.ml,
    wasmThreads: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    cores: nav.hardwareConcurrency ?? 4,
  };
}

/** Backends worth trying on this device, best first. */
export function candidates(c: Capabilities): Backend[] {
  const list: Backend[] = [];
  if (c.webgpu.available) list.push('webgpu');
  if (c.webnn) list.push('webnn');
  list.push('wasm');
  return list;
}

/** Which swap model suits a backend: fp16 halves the download where the GPU computes in half precision. */
export function swapperPrecision(backend: Backend, c: Capabilities): 'fp16' | 'fp32' {
  if (backend === 'webgpu') return c.webgpu.f16 ? 'fp16' : 'fp32';
  if (backend === 'webnn') return 'fp16';
  return 'fp32';
}

/** ONNX Runtime session options for a backend. */
export function sessionOptions(backend: Backend, c: Capabilities) {
  // 'all' fuses Conv+activation into FusedConv, which WebGPU only has in fp32: GPUs get the safe 'basic' level.
  const base = { graphOptimizationLevel: backend === 'wasm' ? 'all' as const : 'basic' as const };
  // NHWC measured 2× faster than NCHW on Apple M-series. ponytail: graph capture gains ~18% more, needs GPU-buffer IO.
  if (backend === 'webgpu') return { ...base, executionProviders: [{ name: 'webgpu' as const, preferredLayout: 'NHWC' as const }] };
  if (backend === 'webnn') return { ...base, executionProviders: [{ name: 'webnn' as const, deviceType: 'gpu' as const, powerPreference: 'high-performance' as const }] };
  return { ...base, executionProviders: ['wasm'] as const, intraOpNumThreads: c.wasmThreads ? Math.max(1, Math.min(8, c.cores - 1)) : 1 };
}

const KEY = 'heisenbert-faceswap:backend';
export function rememberedBackend(): Backend | null {
  try { const v = localStorage.getItem(KEY); return v === 'webgpu' || v === 'webnn' || v === 'wasm' ? v : null; } catch { return null; }
}
export function rememberBackend(b: Backend) {
  try { localStorage.setItem(KEY, b); } catch { /* private mode */ }
}
