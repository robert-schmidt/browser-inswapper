import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { apply, invert, keyPoints, similarity, template, yawOf, type Affine, type Point } from './geometry.ts';
import { FaceFilter, OneEuro, VectorFilter } from './filters.ts';
import { blendLatents, cosine, extractEmap, identityLatent, latentAt, normalize } from './onnx-emap.ts';

const close = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

test('similarity recovers a known rotation, scale and shift', () => {
  const theta = 0.3, s = 1.7, tx = 40, ty = -12;
  const truth: Affine = [s * Math.cos(theta), -s * Math.sin(theta), tx, s * Math.sin(theta), s * Math.cos(theta), ty];
  const src: Point[] = [[10, 20], [80, 22], [45, 60], [20, 95], [70, 96]];
  const dst = src.map((p) => apply(truth, p));
  const m = similarity(src, dst);
  m.forEach((v, i) => close(v, truth[i], 1e-9));
  const back = apply(invert(m), dst[2]);
  close(back[0], src[2][0], 1e-9); close(back[1], src[2][1], 1e-9);
});

test('templates match the reference models (112 as is, 128 shifted by 8)', () => {
  assert.deepEqual(template(112)[0], [38.2946, 51.6963]);
  const t = template(128);
  close(t[0][0], 46.2946); close(t[0][1], 51.6963);
});

test('key points come out image-left first, mirrored or not', () => {
  const lm = new Float32Array(478 * 3);
  const set = (i: number, x: number, y: number) => { lm[i * 3] = x; lm[i * 3 + 1] = y; };
  set(468, 0.7, 0.4); set(33, 0.72, 0.4); set(133, 0.66, 0.4); // subject's right eye on the image right (mirrored)
  set(473, 0.3, 0.4); set(362, 0.34, 0.4); set(263, 0.28, 0.4);
  set(1, 0.5, 0.55); set(61, 0.62, 0.7); set(291, 0.38, 0.7);
  const k = keyPoints(lm, 100, 100);
  assert.ok(k[0][0] < k[1][0] && k[3][0] < k[4][0]);
});

test('One-Euro filter settles on a still value and follows a jump', () => {
  const f = new OneEuro();
  let v = 0;
  for (let t = 0; t < 1000; t += 33) v = f.filter(10 + (t % 66 ? 0.2 : -0.2), t);
  close(v, 10, 0.25);
  for (let t = 1000; t < 1600; t += 33) v = f.filter(50, t);
  close(v, 50, 1);
});

test('identity latent is unit length; blends stay on the sphere and move between ends', () => {
  const emap = new Float32Array(512 * 512);
  for (let i = 0; i < 512; i++) emap[i * 512 + i] = 2; // scaled identity
  const a = identityLatent(Float32Array.from({ length: 512 }, (_, i) => Math.sin(i)), emap);
  const b = identityLatent(Float32Array.from({ length: 512 }, (_, i) => Math.cos(i)), emap);
  close(Math.hypot(...a), 1, 1e-5);
  const mid = blendLatents(a, b, 0.5);
  close(Math.hypot(...mid), 1, 1e-5);
  assert.ok(cosine(mid, a) > cosine(b, a) && cosine(mid, b) > cosine(a, b));
  close(cosine(blendLatents(a, b, 0), a), 1, 1e-6);
  void normalize;
});

const MODEL = new URL('../models/inswapper_128_fp16.onnx', import.meta.url);
const REF = new URL('../models/emap-reference.bin', import.meta.url);
test('emap read from the model bytes equals the onnx library reference', { skip: !existsSync(MODEL) || !existsSync(REF) }, () => {
  const emap = extractEmap(readFileSync(MODEL));
  const r = readFileSync(REF);
  const ref = new Float32Array(r.buffer.slice(r.byteOffset, r.byteOffset + r.byteLength));
  assert.equal(emap.length, ref.length);
  for (let i = 0; i < ref.length; i += 997) assert.equal(emap[i], ref[i]);
  assert.equal(emap[0], ref[0]); assert.equal(emap[ref.length - 1], ref[ref.length - 1]);
});

test('mesh: triangles from edges, opacity fades from the contour inwards', async () => {
  const { buildMesh } = await import('./mesh.ts');
  // A square split by a diagonal, plus a centre vertex 4 joined to all corners.
  const e = (start: number, end: number) => ({ start, end });
  const { triangles, weights } = buildMesh([e(0, 1), e(1, 2), e(2, 3), e(3, 0), e(0, 4), e(1, 4), e(2, 4), e(3, 4)], [0, 1, 2, 3], 5, [0, 0.5]);
  assert.equal(triangles.length / 3, 4);
  assert.deepEqual([...weights], [0, 0, 0, 0, 0.5]);
});

test('photo checks: a sharp, frontal, well-lit face has no issues; blur, size and light are caught', async () => {
  const { faceIssues, sharpness } = await import('./faces.ts');
  const pts = new Float32Array(478 * 2);
  const set = (i: number, x: number, y: number) => { pts[i * 2] = x; pts[i * 2 + 1] = y; };
  set(33, 100, 100); set(263, 200, 100); set(234, 60, 140); set(454, 240, 140); set(1, 150, 150);
  const checker = Array.from({ length: 112 * 112 }, (_, i) => (((i % 112) >> 2) + ((i / 112) >> 2)) % 2 ? 180 : 60);
  assert.deepEqual(faceIssues(pts, checker, 112), []);
  const flat = new Array(112 * 112).fill(120);
  assert.ok(sharpness(flat, 112) < 1);
  assert.deepEqual(faceIssues(pts, flat, 112), ['blurry']);
  assert.deepEqual(faceIssues(pts, flat.map(() => 20), 112), ['dark', 'blurry']);
  set(263, 130, 100);
  assert.ok(faceIssues(pts, checker, 112).includes('small'));
  set(263, 200, 100); set(1, 210, 150);
  assert.ok(faceIssues(pts, checker, 112).includes('turned'));
});

test('yaw: read from a pose matrix, unaffected by tilt and nod', () => {
  // Column-major 4×4 of R = Rz(roll)·Ry(yaw)·Rx(pitch), as MediaPipe gives it.
  const pose = (yaw: number, roll: number, pitch: number) => {
    const [y, r, p] = [yaw, roll, pitch].map((d) => (d * Math.PI) / 180);
    const mul = (A: number[][], B: number[][]) => A.map((row) => B[0].map((_, c) => row.reduce((s, v, k) => s + v * B[k][c], 0)));
    const Rz = [[Math.cos(r), -Math.sin(r), 0], [Math.sin(r), Math.cos(r), 0], [0, 0, 1]];
    const Ry = [[Math.cos(y), 0, Math.sin(y)], [0, 1, 0], [-Math.sin(y), 0, Math.cos(y)]];
    const Rx = [[1, 0, 0], [0, Math.cos(p), -Math.sin(p)], [0, Math.sin(p), Math.cos(p)]];
    const R = mul(mul(Rz, Ry), Rx), m = new Float32Array(16);
    for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) m[col * 4 + row] = R[row][col];
    m[15] = 1;
    return m;
  };
  for (const yaw of [-40, 0, 25]) { close(yawOf(pose(yaw, 0, 0)), yaw, 1e-4); close(yawOf(pose(yaw, 20, -15)), yaw, 1e-4); }
});

test('pose blend: matching photo at its angle, halfway between two, clamped past the ends', () => {
  const unit = (i: number) => { const v = new Float32Array(512); v[i] = 1; return v; };
  const views = [{ latent: unit(0), yaw: -30 }, { latent: unit(1), yaw: 0 }, { latent: unit(2), yaw: 28 }];
  assert.deepEqual(latentAt(views, 0), views[1].latent);
  assert.equal(latentAt(views, -70), views[0].latent);
  assert.equal(latentAt(views, 60), views[2].latent);
  const mid = latentAt(views, 14);
  close(mid[1], mid[2], 1e-6); close(mid[0], 0);
  assert.equal(latentAt([views[1]], 45), views[1].latent);
});

test('mesh: triangles wind alike; a folded-over side is found and its silhouette fades', async () => {
  const { buildMesh, visibility } = await import('./mesh.ts');
  // A 4×3 strip of squares (vertices 0..19, row-major 5 wide), outline = the border.
  const W = 5, H = 4, e = (a: number, b: number) => ({ start: a, end: b }), edges = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = y * W + x;
    if (x < W - 1) edges.push(e(v, v + 1));
    if (y < H - 1) edges.push(e(v, v + W));
    if (x < W - 1 && y < H - 1) edges.push(e(v, v + W + 1));
  }
  const border = [...Array(W * H).keys()].filter((v) => v % W === 0 || v % W === W - 1 || v < W || v >= W * (H - 1));
  const mesh = buildMesh(edges, border, W * H, [1]);
  const flat = new Float32Array(W * H * 2);
  for (let v = 0; v < W * H; v++) { flat[v * 2] = v % W; flat[v * 2 + 1] = Math.floor(v / W); }
  const facing = visibility(mesh, flat);
  assert.ok(facing.away.every((a) => a === 0), 'flat: every triangle faces the same way');
  assert.deepEqual([...facing.weights], [...mesh.weights], 'flat: nothing fades');
  // Fold the right-hand column back over its neighbour (a head turned far: the far side lands behind).
  const folded = flat.slice();
  for (let y = 0; y < H; y++) folded[(y * W + 4) * 2] = 2.5;
  const f = visibility(mesh, folded);
  assert.equal(f.away.filter(Boolean).length, 6, 'the folded column faces away');
  for (let y = 0; y < H; y++) { assert.equal(f.weights[y * W + 3], 0, 'fold line'); }
  assert.equal(f.weights[1 * W + 2], 0.5, 'one step in');
  assert.equal(f.weights[1 * W + 1], 1, 'further in');
});

test('face filter: keeps up with a fast moving, tilting head; still face as steady as per-point smoothing', () => {
  let seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const noise = () => (rnd() + rnd() + rnd() - 1.5) * 2; // ~1 px
  const shape = Array.from({ length: 120 }, () => [(rnd() - 0.5) * 200, (rnd() - 0.5) * 260]);
  const score = (f: { filter(v: number[], t: number): ArrayLike<number> }) => {
    let lag = 0, jitter = 0, prev: ArrayLike<number> | null = null;
    for (let k = 0; k < 60; k++) {
      const s = k / 30, m = s < 1 ? 0 : s < 1.4 ? s - 1 : 0.4, a = m * 2; // 1000 px/s sideways, ~115°/s tilt
      const truth = shape.flatMap(([x, y]) => [640 + m * 1000 + x * Math.cos(a) - y * Math.sin(a), 360 + x * Math.sin(a) + y * Math.cos(a)]);
      const out = f.filter(truth.map((v) => v + noise()), k * 1000 / 30);
      if (s >= 1) { let d = 0; for (let i = 0; i < out.length; i += 2) d += Math.hypot(out[i] - truth[i], out[i + 1] - truth[i + 1]); lag = Math.max(lag, d / shape.length); }
      if (s > 0.3 && s < 1 && prev) { let d = 0; for (let i = 0; i < out.length; i++) d += (out[i] - prev[i]) ** 2; jitter += Math.sqrt(d / shape.length); }
      prev = out;
    }
    return { lag, jitter };
  };
  const old = score(new VectorFilter(3.0, 0.02)), now = score(new FaceFilter());
  assert.ok(now.lag < old.lag / 3, `lag ${now.lag.toFixed(1)} px vs ${old.lag.toFixed(1)} px`);
  assert.ok(now.lag < 4, `lag ${now.lag.toFixed(1)} px`);
  assert.ok(now.jitter < old.jitter * 1.25, `jitter ${now.jitter.toFixed(2)} vs ${old.jitter.toFixed(2)}`);
});
