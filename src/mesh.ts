// The face mesh the swapped face is drawn on: triangles recovered from MediaPipe's tessellation edges, and a
// per-vertex opacity that fades out towards the face contour (a soft edge that follows the jaw and hairline).
// Triangles all wind the same way, so the ones facing away from the camera (the far side of a turned head,
// folded over the near side) can be told apart and left out.

/** `contour`: 1 for vertices on the face outline. */
export type Mesh = { triangles: Uint16Array; weights: Float32Array; contour: Uint8Array };

/**
 * `edges`: FaceLandmarker.FACE_LANDMARKS_TESSELATION. `contour`: the outline (FACE_OVAL). `fade`: opacity by ring.
 * `holes`: openings to fill, each from a centre point to the ring of points around it (the eyes: EYES).
 */
export function buildMesh(edges: { start: number; end: number }[], contour: number[], vertices = 478, fade = [0.1, 0.75], holes: { center: number; rim: number[] }[] = []): Mesh {
  const adj = Array.from({ length: vertices }, () => new Set<number>());
  for (const { start: a, end: b } of edges) { adj[a].add(b); adj[b].add(a); }
  // Every 3-cycle of the planar mesh is a face triangle (the mesh has no separating triangles worth worrying about).
  const tris: number[] = [];
  for (let a = 0; a < vertices; a++) {
    for (const b of adj[a]) {
      if (b <= a) continue;
      for (const c of adj[b]) if (c > b && adj[a].has(c)) tris.push(a, b, c);
    }
  }
  for (const { center, rim } of holes) {
    rim.forEach((v, i) => {
      const w = rim[(i + 1) % rim.length];
      tris.push(center, v, w);
      adj[center].add(v); adj[v].add(center);
    });
  }
  orient(tris);
  // Rings inward from the contour, by graph distance.
  const ring = new Int32Array(vertices).fill(-1);
  let front = contour.filter((i) => i < vertices);
  for (const i of front) ring[i] = 0;
  for (let r = 1; front.length && r < fade.length; r++) {
    const next: number[] = [];
    for (const i of front) for (const j of adj[i]) if (ring[j] === -1) { ring[j] = r; next.push(j); }
    front = next;
  }
  const weights = new Float32Array(vertices);
  for (let i = 0; i < vertices; i++) weights[i] = ring[i] === -1 ? 1 : fade[ring[i]];
  const onContour = new Uint8Array(vertices);
  for (const i of contour) if (i < vertices) onContour[i] = 1;
  return { triangles: new Uint16Array(tris), weights, contour: onContour };
}

/** Makes every triangle wind like its neighbours (a shared edge runs opposite ways in the two), in place. */
function orient(tris: number[]) {
  const n = tris.length / 3, byEdge = new Map<string, number[]>(), done = new Uint8Array(n);
  const key = (a: number, b: number) => (a < b ? `${a},${b}` : `${b},${a}`);
  for (let t = 0; t < n; t++) for (let k = 0; k < 3; k++) {
    const e = key(tris[t * 3 + k], tris[t * 3 + ((k + 1) % 3)]);
    byEdge.set(e, [...(byEdge.get(e) ?? []), t]);
  }
  const runs = (t: number, a: number, b: number) => [0, 1, 2].some((k) => tris[t * 3 + k] === a && tris[t * 3 + ((k + 1) % 3)] === b);
  for (let start = 0; start < n; start++) {
    if (done[start]) continue;
    done[start] = 1;
    const queue = [start];
    while (queue.length) {
      const t = queue.pop()!;
      for (let k = 0; k < 3; k++) {
        const a = tris[t * 3 + k], b = tris[t * 3 + ((k + 1) % 3)];
        for (const u of byEdge.get(key(a, b))!) {
          if (done[u]) continue;
          if (runs(u, a, b)) [tris[u * 3 + 1], tris[u * 3 + 2]] = [tris[u * 3 + 2], tris[u * 3 + 1]];
          done[u] = 1;
          queue.push(u);
        }
      }
    }
  }
}

/** Twice the signed area of each triangle at `points` (x/y pairs). */
function areas(mesh: Mesh, points: ArrayLike<number>) {
  const t = mesh.triangles, out = new Float32Array(t.length / 3);
  for (let i = 0; i < out.length; i++) {
    const a = t[i * 3] * 2, b = t[i * 3 + 1] * 2, c = t[i * 3 + 2] * 2;
    out[i] = (points[b] - points[a]) * (points[c + 1] - points[a + 1]) - (points[b + 1] - points[a + 1]) * (points[c] - points[a]);
  }
  return out;
}

/**
 * Which way the visible side of the face winds at `points` (+1 or -1: the sign most of its area has), which
 * triangles face away, and the opacity per vertex: as the mesh's, but 0 along the silhouette of a turned head
 * (where the far side folds away behind it) and half one step in, so the face fades out there instead of ending
 * in a hard edge. Hidden pockets inside the face (behind the nose at a small turn) don't fade anything.
 */
export function visibility(mesh: Mesh, points: ArrayLike<number>) {
  const area = areas(mesh, points), t = mesh.triangles, n = area.length;
  const front = area.reduce((s, a) => s + a, 0) >= 0 ? 1 : -1;
  const away = area.map((a) => (a * front < 0 ? 1 : 0));
  // The far side: turned-away triangles connected to the face outline.
  const far = new Uint8Array(mesh.weights.length);
  let grew = true;
  for (let i = 0; i < n; i++) if (away[i] && (mesh.contour[t[i * 3]] || mesh.contour[t[i * 3 + 1]] || mesh.contour[t[i * 3 + 2]])) far[t[i * 3]] = far[t[i * 3 + 1]] = far[t[i * 3 + 2]] = 1;
  while (grew) {
    grew = false;
    for (let i = 0; i < n; i++) {
      if (!away[i]) continue;
      const [a, b, c] = [t[i * 3], t[i * 3 + 1], t[i * 3 + 2]];
      if ((far[a] || far[b] || far[c]) && !(far[a] && far[b] && far[c])) { far[a] = far[b] = far[c] = 1; grew = true; }
    }
  }
  const weights = mesh.weights.slice();
  for (let i = 0; i < n; i++) {
    if (away[i]) continue;
    const v = [t[i * 3], t[i * 3 + 1], t[i * 3 + 2]];
    if (!v.some((x) => far[x])) continue;
    for (const x of v) weights[x] = far[x] ? 0 : Math.min(weights[x], 0.5);
  }
  return { front, away, weights };
}
