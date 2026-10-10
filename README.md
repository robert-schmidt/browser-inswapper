# heisenbert-faceswap

Real-time face swap that runs entirely in the browser: no server, nothing uploaded. Give it a camera track
and a face, get back a swapped `MediaStreamTrack` you can show, record or stream. Switch faces while it runs
and it morphs smoothly from one to the next.

Built for [heisenbert.com](https://heisenbert.com), the AI co-host for streams and recordings.

**[Try the live demo](https://robert-schmidt.github.io/heisenbert-faceswap/)**: it runs in your browser and nothing is uploaded.
Use your camera or the sample clip, swap faces, morph between them, and record a clip to download.

> **Free for non-commercial use** under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).
> To use it in a commercial product or service, email **robert@chronic.ro** first.

## What it does

- **Runs on whatever the device has.** On first use it benchmarks this device once and remembers the fastest option:
  WebGPU (Metal on Apple Silicon, D3D12 or Vulkan on Windows and Linux GPUs), WebNN where the browser has it,
  or multi-threaded WebAssembly on the CPU.
- **Picks the right model precision.** It uses half-precision weights on GPUs that support `shader-f16`
  (half the download, faster) and full precision everywhere else.
- **Stays smooth when the model is slower than the camera.** The swap model runs as often as the device allows,
  for example about 10 times a second on an Apple M-series GPU. In between, the latest swapped face is drawn on a
  478-point face mesh that follows the current frame, so head turns, the mouth, eyes and brows move at the
  camera's full frame rate. Each new result crossfades in over 70 ms, and its colour correction is smoothed
  over time, so there's no shimmer.
- **Keeps hands in front.** A segmentation model finds face skin, so a hand, microphone or cup in front of the
  face stays in front instead of being painted over. It runs in its own worker and never holds up the video.
- **Checks photos.** `findFaces` finds every face in a photo, including small faces in group or full-body
  shots, and flags what would make the result less accurate: a small, turned, blurry, dark or overexposed face.
- **Morphs live.** Switching identity blends between the two faces over the time you choose, while recording or streaming.
- **Fades in and out** without losing the identity (`setStrength`).
- **Keeps the page responsive.** The engine runs in a Web Worker, and late frames are dropped instead of queued,
  so latency never builds up.
- **Works in any framework**, with optional React hooks.
- **Downloads models once.** They're kept in Cache Storage and checked against a SHA-256 before use.

## Install

```sh
npm install heisenbert-faceswap
```

## Quick start

```ts
import { createFaceSwap, KNOWN_SHA256 } from 'heisenbert-faceswap';

const base = 'https://your-cdn.example/models/';
const swap = await createFaceSwap({
  models: {
    swapperFp16: { url: base + 'inswapper_128_fp16.onnx', sha256: KNOWN_SHA256.swapperFp16 },
    swapperFp32: { url: base + 'inswapper_128.onnx', sha256: KNOWN_SHA256.swapperFp32 },
    arcface: { url: base + 'w600k_r50.onnx', sha256: KNOWN_SHA256.arcface },
    // landmarker: defaults to Google's hosted MediaPipe Face Landmarker
  },
  onProgress: ({ model, loaded, total }) => console.log(model, loaded, total),
  onStats: ({ fps, inferMs, backend }) => console.log(backend, fps, inferMs),
});

const camera = (await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })).getVideoTracks()[0];
const swapped = swap.process(camera);                  // a MediaStreamTrack
video.srcObject = new MediaStream([swapped]);          // or MediaRecorder, or RTCPeerConnection.addTrack

const [face] = await swap.findFaces(photo);            // biggest first; face.issues says what's off, face.preview shows it
const embedding = await swap.setIdentity({ image: photo, face });
// Save `embedding` (512 numbers) and pass { embedding } next time: no identity model download needed.

swap.setIdentity({ image: '/faces/other.png' }, { morphMs: 1500 });  // morph while running
swap.setStrength(0, 400);                                            // fade back to the real face
swap.dispose();
```

### React (Next.js, Remix, Vite…)

```tsx
'use client';
import { useMemo, useState } from 'react';
import { useFaceSwap, useSwappedTrack, TrackVideo } from 'heisenbert-faceswap/react';

export function Mirror({ camera }: { camera: MediaStreamTrack }) {
  const options = useMemo(() => ({ models }), []);
  const { status, progress, swap, stats } = useFaceSwap(options);
  const [identity, setIdentity] = useState<{ image: string } | null>({ image: '/faces/a.png' });
  const track = useSwappedTrack(swap, camera, identity, 1500);
  if (status !== 'ready') return <p>Loading {progress?.model}…</p>;
  return <TrackVideo track={track} />;
}
```

The library only runs in the browser. In server-rendered frameworks, use it in client components.

### Your own render loop

`swap.processFrame(source)` takes a video element, canvas, `ImageBitmap` or `VideoFrame` and resolves to the
swapped `ImageBitmap`.

## Performance

Each swap is about 349 billion operations (the model's 128×128 face generator). Output video always keeps
the camera's frame rate; the faster the GPU, the more often the face itself is freshly redrawn.

| Device | Runs on | Per swap | Fresh faces per second | |
| --- | --- | --- | --- | --- |
| MacBook Pro, M2 Max (38-core GPU) | WebGPU (Metal) | 95 ms | ~10 | measured |
| M1 / M2 (8–10-core GPU) | WebGPU (Metal) | 400–550 ms | 2–3 | estimate |
| M4 Pro (20-core GPU) | WebGPU (Metal) | 110–150 ms | 7–9 | estimate |
| M3 Max / M4 Max (40-core GPU) | WebGPU (Metal) | 60–80 ms | 13–16 | estimate |
| RTX 3060 | WebGPU (D3D12 / Vulkan) | 90–140 ms | 7–11 | estimate |
| RTX 3090 | WebGPU (D3D12 / Vulkan) | 35–50 ms | 20–28 | estimate |
| RTX 4070 | WebGPU (D3D12 / Vulkan) | 40–60 ms | 17–25 | estimate |
| RTX 4090 | WebGPU (D3D12 / Vulkan) | 15–25 ms | 30 (every frame) | estimate |
| RTX 5090 | WebGPU (D3D12 / Vulkan) | 12–20 ms | 30 (every frame) | estimate |
| Intel / AMD integrated graphics | WebGPU | 600–1,000 ms | ~1 | estimate |
| Any computer without WebGPU | WebAssembly (CPU) | 5–15 s | not usable live | measured on M2 Max |

Estimates scale the M2 Max measurement by each GPU's FP32 throughput, assuming WebGPU reaches 20–35% of peak
on this model (the M2 Max reaches about 27%). Treat them as a range, and check your own device with the demo:
it shows fps, swaps per second and latency.

Other numbers from the M2 Max:
- **Latency:** 35–40 ms from a camera frame arriving to its swapped copy leaving (tracking and compositing;
  the face shown is at most one swap older).
- **Output:** 27–30 fps with occlusion on.
- **Startup:** the first start downloads about 300 MB; later starts load from the browser cache in 1–2 s.

## Browser support

| Browser | Inference | Video path |
| --- | --- | --- |
| Chrome, Edge (desktop) | WebGPU, WebNN (behind `chrome://flags/#web-machine-learning-neural-network`), WASM | `MediaStreamTrackProcessor` (zero-copy `VideoFrame`s) |
| Safari 18+ | WebGPU, WASM | canvas `captureStream` |
| Firefox | WebGPU where enabled, otherwise WASM | canvas `captureStream` |

For multi-threaded CPU inference, serve the page cross-origin isolated:
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` (or `require-corp`).

## Models and their licences

No model weights ship with this package. You host them, and their licences are separate from this library's:

| Model | Used for | Licence |
| --- | --- | --- |
| `inswapper_128` (fp32 or fp16) | The swap itself | By InsightFace. Non-commercial research use; commercial licences from InsightFace |
| `w600k_r50` (ArcFace, from InsightFace `buffalo_l`) | Turning a photo into an identity | By InsightFace. Non-commercial research use; commercial licences from InsightFace |
| MediaPipe Face Landmarker | Finding and tracking the face (478 points) | Apache-2.0, by Google |
| MediaPipe multiclass selfie segmenter | Keeping hands and objects in front of the face (`occlusion`, on by default) | Apache-2.0, by Google |

The MediaPipe models default to Google's hosted copies. The hosted demo loads the InsightFace models from public
Hugging Face mirrors whose files match these fingerprints. `KNOWN_SHA256` lists the fingerprints of the public builds this library was developed and tested with.
Identities can be stored as embeddings, so apps can offer preset faces without ever loading the ArcFace model.

## How it works

1. **Find the face.** MediaPipe Face Landmarker tracks 478 face points on each frame, on the GPU where available.
2. **Align.** Five key points (eye centres, nose tip, mouth corners) map the face onto the model's 128×128
   template with a least-squares similarity transform. Landmarks are lightly smoothed with a One-Euro filter,
   which steadies a still face without adding lag.
3. **Swap.** The model redraws the aligned face with the target identity: a 512-number identity vector
   (ArcFace, projected through the model's own embedding map). Morphing interpolates between two identity
   vectors on the sphere.
4. **Blend.** The new face is colour-matched to the scene lighting. It is then drawn with WebGL on the face
   mesh of the current frame, with a soft edge that follows the jaw and hairline, only where the segmenter sees
   face skin.

## Run the demo

```sh
npm install
npm run demo        # dev server on http://localhost:5180 (put the model files in ./models first)
```

The demo has a built-in sample video, so it works without a camera, and runs the swap on three fictional,
AI-generated faces. Pushes to `main` publish it to GitHub Pages (`.github/workflows/pages.yml`).

GitHub Pages can't send cross-origin isolation headers, so the hosted demo runs CPU inference on one thread.
That only matters on computers without WebGPU.

## Use it responsibly

Swap only faces you have the right to use (your own, or characters made for you), and label altered video as
altered, especially in live streams. Don't use this library to impersonate real people, deceive viewers, or
make sexual or harassing content. Platform rules (for example YouTube's altered-content disclosure) apply to
what you publish.

## Licence

Copyright Chronic Qual SRL. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md);
see [NOTICE.md](NOTICE.md). For commercial licensing, email robert@chronic.ro.
