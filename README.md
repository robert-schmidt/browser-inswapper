# heisenbert-faceswap

Real-time face swap that runs entirely in the browser: no server, nothing uploaded. Give it a camera track
and a face, get back a swapped `MediaStreamTrack` you can show, record or stream. Switch faces while it runs
and it morphs smoothly from one to the next.

Built for [heisenbert.com](https://heisenbert.com), the AI co-host for streams and recordings.

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
  camera's full frame rate.
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

const embedding = await swap.setIdentity({ image: '/faces/character.png' });
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

## Browser support

| Browser | Inference | Video path |
| --- | --- | --- |
| Chrome, Edge (desktop) | WebGPU, WebNN (where enabled), WASM | `MediaStreamTrackProcessor` (zero-copy `VideoFrame`s) |
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

`KNOWN_SHA256` lists the fingerprints of the public builds this library was developed and tested with.
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
   mesh of the current frame, with a soft edge that follows the jaw and hairline.

## Run the demo

```sh
npm install
npm run demo        # dev server on http://localhost:5180 (put the model files in ./models first)
```

The demo has a built-in sample video, so it works without a camera, and runs the swap on three fictional,
AI-generated faces.

## Use it responsibly

Swap only faces you have the right to use (your own, or characters made for you), and label altered video as
altered, especially in live streams. Don't use this library to impersonate real people, deceive viewers, or
make sexual or harassing content. Platform rules (for example YouTube's altered-content disclosure) apply to
what you publish.

## Licence

Copyright Chronic Qual SRL. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md);
see [NOTICE.md](NOTICE.md). For commercial licensing, email robert@chronic.ro.
