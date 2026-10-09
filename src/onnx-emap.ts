// The swap model projects the source identity through a 512×512 matrix ("emap") stored as its last
// initializer. ONNX Runtime doesn't expose initializers, so we read it straight from the model bytes:
// a tiny protobuf walk (ModelProto.graph → GraphProto.initializer → TensorProto), no library, no copy.

function varint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0, shift = 0, b: number;
  do {
    b = buf[pos++];
    result += (b & 0x7f) * 2 ** shift; // multiply, not <<: lengths can pass 2^31
    shift += 7;
  } while (b & 0x80);
  return [result, pos];
}

type Field = { field: number; wire: number; start: number; end: number; value: number };
function* fields(buf: Uint8Array, start: number, end: number): Generator<Field> {
  let pos = start;
  while (pos < end) {
    let key: number;
    [key, pos] = varint(buf, pos);
    const field = Math.floor(key / 8), wire = key & 7;
    if (wire === 0) { const [v, p] = varint(buf, pos); yield { field, wire, start: pos, end: p, value: v }; pos = p; }
    else if (wire === 2) { const [len, p] = varint(buf, pos); yield { field, wire, start: p, end: p + len, value: len }; pos = p + len; }
    else if (wire === 5) { yield { field, wire, start: pos, end: pos + 4, value: 0 }; pos += 4; }
    else if (wire === 1) { yield { field, wire, start: pos, end: pos + 8, value: 0 }; pos += 8; }
    else throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
}

/** Reads the last 512×512 float initializer (the identity projection) from an inswapper ONNX model. */
export function extractEmap(model: ArrayBuffer | Uint8Array): Float32Array {
  const buf = model instanceof Uint8Array ? model : new Uint8Array(model);
  let graph: Field | undefined;
  for (const f of fields(buf, 0, buf.length)) if (f.field === 7 && f.wire === 2) graph = f;
  if (!graph) throw new Error('Not an ONNX model (no graph).');
  let emap: Float32Array | null = null;
  for (const init of fields(buf, graph.start, graph.end)) {
    if (init.field !== 5 || init.wire !== 2) continue; // GraphProto.initializer
    const dims: number[] = [];
    let type = 0, raw: Field | undefined, floats: Field | undefined;
    for (const t of fields(buf, init.start, init.end)) {
      if (t.field === 1 && t.wire === 0) dims.push(t.value);
      else if (t.field === 1 && t.wire === 2) for (let p = t.start; p < t.end;) { const [v, n] = varint(buf, p); dims.push(v); p = n; } // packed dims
      else if (t.field === 2) type = t.value;
      else if (t.field === 9) raw = t;
      else if (t.field === 4 && t.wire === 2) floats = t;
    }
    if (dims.length !== 2 || dims[0] !== 512 || dims[1] !== 512) continue;
    const src = raw ?? floats;
    if (!src) continue;
    if (type === 1) { // FLOAT, little-endian
      emap = new Float32Array(new Uint8Array(buf.subarray(src.start, src.end)).buffer); // copy: aligned, and doesn't pin the model
    } else if (type === 10) { // FLOAT16
      const h = new Uint16Array(new Uint8Array(buf.subarray(src.start, src.end)).buffer);
      emap = Float32Array.from(h, halfToFloat);
    }
  }
  if (!emap || emap.length !== 512 * 512) throw new Error('No 512×512 identity projection found in the swap model.');
  return emap;
}

function halfToFloat(h: number) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

/** Source identity → the swap model's input: normalise the ArcFace embedding, project by emap, normalise. */
export function identityLatent(embedding: Float32Array, emap: Float32Array): Float32Array {
  const e = normalize(embedding);
  const out = new Float32Array(512);
  for (let j = 0; j < 512; j++) {
    let s = 0;
    for (let i = 0; i < 512; i++) s += e[i] * emap[i * 512 + j];
    out[j] = s;
  }
  return normalize(out);
}

export function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

/** Identity morph: spherical-ish blend of two latents (lerp + renormalise keeps the result on the sphere). */
export function blendLatents(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * (1 - t) + b[i] * t;
  return normalize(out);
}

export function cosine(a: Float32Array, b: Float32Array) {
  const x = normalize(a), y = normalize(b);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * y[i];
  return s;
}
