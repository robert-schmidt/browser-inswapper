import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { apply, invert, keyPoints, similarity, template, type Affine, type Point } from './geometry.ts';
import { OneEuro } from './filters.ts';
import { blendLatents, cosine, extractEmap, identityLatent, normalize } from './onnx-emap.ts';

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
