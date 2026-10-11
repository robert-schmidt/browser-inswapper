// Geometry for face alignment: the 5 key points the swap and identity models were trained on, the
// least-squares similarity transform onto them, and 2×3 affine helpers (canvas-compatible).

export type Point = [number, number];
/** Row-major 2×3 affine: x' = a·x + b·y + c, y' = d·x + e·y + f. */
export type Affine = [number, number, number, number, number, number];

/** ArcFace's canonical 5 points in a 112×112 crop: left eye, right eye, nose, left and right mouth corner (image left/right). */
export const ARCFACE_112: Point[] = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];

/** The template for a square crop of `size` px, as the reference models were trained: 112 → as is, 128 → shifted 8 px right. */
export function template(size: number): Point[] {
  const ratio = size % 112 === 0 ? size / 112 : size / 128;
  const dx = size % 112 === 0 ? 0 : 8 * ratio;
  return ARCFACE_112.map(([x, y]) => [x * ratio + dx, y * ratio]);
}

/** Least-squares similarity (rotation, uniform scale, translation; no reflection) taking `src` onto `dst`. */
export function similarity(src: Point[], dst: Point[]): Affine {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const px = src[i][0] - sx, py = src[i][1] - sy, qx = dst[i][0] - dx, qy = dst[i][1] - dy;
    num1 += px * qx + py * qy;
    num2 += px * qy - py * qx;
    den += px * px + py * py;
  }
  const a = den ? num1 / den : 1, b = den ? num2 / den : 0;
  return [a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy)];
}

export function invert([a, b, c, d, e, f]: Affine): Affine {
  const det = a * e - b * d;
  return [e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det];
}

export const apply = ([a, b, c, d, e, f]: Affine, [x, y]: Point): Point => [a * x + b * y + c, d * x + e * y + f];

/** Arguments for CanvasRenderingContext2D.setTransform (a, b, c, d, e, f in canvas order). */
export const toCanvas = ([a, b, c, d, e, f]: Affine): [number, number, number, number, number, number] => [a, d, b, e, c, f];

// MediaPipe Face Landmarker (478 points, with irises) → the 5 key points. Pairs are sorted by x so a mirrored
// camera still gives "image left" first, as the models expect.
const IRIS = [468, 473], EYE_CORNERS = [[33, 133], [362, 263]], NOSE_TIP = 1, MOUTH_CORNERS = [61, 291];
export function keyPoints(lm: ArrayLike<number>, width: number, height: number, stride = 3): Point[] {
  const p = (i: number): Point => [lm[i * stride] * width, lm[i * stride + 1] * height];
  const mid = (i: number, j: number): Point => { const a = p(i), b = p(j); return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; };
  const hasIris = lm.length >= 478 * stride;
  // Eye centre: the iris when tracked (what detectors call the eye point), blended with the corners for stability.
  const eye = (k: 0 | 1): Point => {
    const c = mid(EYE_CORNERS[k][0], EYE_CORNERS[k][1]);
    if (!hasIris) return c;
    const iris = p(IRIS[k]);
    return [iris[0] * 0.6 + c[0] * 0.4, iris[1] * 0.6 + c[1] * 0.4];
  };
  const eyes = [eye(0), eye(1)].sort((u, v) => u[0] - v[0]);
  const mouth = [p(MOUTH_CORNERS[0]), p(MOUTH_CORNERS[1])].sort((u, v) => u[0] - v[0]);
  return [eyes[0], eyes[1], p(NOSE_TIP), mouth[0], mouth[1]];
}

/**
 * The eye openings: each iris centre and the eyelid rim around it, in order. MediaPipe's mesh leaves the openings
 * empty; filled from the iris centre, the swapped eyes are drawn too, and follow the gaze on every frame.
 */
export const EYES = [
  { center: 468, rim: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7] },
  { center: 473, rim: [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249] },
];

/** The mouth opening, inner lip edges from corner to corner: filled, the swapped teeth move with the swapped lips. */
export const MOUTH = {
  upper: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  lower: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
};

/** MediaPipe's face-oval contour, in order (for the paste-back mask). */
export const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

/**
 * Head turn in degrees from MediaPipe's facial transformation matrix (4×4, column-major: the canonical face model
 * fitted to the landmarks): 0 facing the camera, positive with the nose towards the image's right. A tilted or
 * nodding head doesn't change it. (Measured on the landmarks' own depth instead, small turns read about a
 * third of this and noisier.)
 */
export function yawOf(matrix: ArrayLike<number>): number {
  // The head's left-right axis (first column) leaves the image plane by sin(yaw), whatever the tilt or nod.
  return (Math.asin(Math.max(-1, Math.min(1, -matrix[2]))) * 180) / Math.PI;
}
