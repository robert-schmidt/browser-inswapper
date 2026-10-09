// The face mesh the swapped face is drawn on: triangles recovered from MediaPipe's tessellation edges, and a
// per-vertex opacity that fades out towards the face contour (a soft edge that follows the jaw and hairline).

export type Mesh = { triangles: Uint16Array; weights: Float32Array };

/** `edges`: FaceLandmarker.FACE_LANDMARKS_TESSELATION. `contour`: the outline (FACE_OVAL). `fade`: opacity by ring. */
export function buildMesh(edges: { start: number; end: number }[], contour: number[], vertices = 478, fade = [0, 0.55, 0.9]): Mesh {
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
  return { triangles: new Uint16Array(tris), weights };
}
