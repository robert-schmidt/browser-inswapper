// Checks on a photo used as an identity: what would make the swapped face look less like the person.
// ponytail: thresholds are hand-tuned heuristics on typical phone and webcam photos; adjust with real uploads.

export type FaceIssue = 'small' | 'turned' | 'blurry' | 'dark' | 'bright';

export const ISSUE_TEXT: Record<FaceIssue, string> = {
  small: 'The face is small in this photo, so the result may look soft. A closer photo works better.',
  turned: 'The face is turned to the side. A photo looking at the camera works best.',
  blurry: 'The photo looks blurry. A sharper one gives a more accurate face.',
  dark: 'The face is dark. A brighter photo gives a more accurate face.',
  bright: 'The face is overexposed. A photo with softer light works better.',
};

/** `points`: landmarks in pixels (x, y pairs). `gray`: the aligned face crop's luminance (0–255). */
export function faceIssues(points: ArrayLike<number>, gray: Uint8ClampedArray | number[], size: number): FaceIssue[] {
  const p = (i: number) => [points[i * 2], points[i * 2 + 1]];
  const issues: FaceIssue[] = [];
  const [l, r] = [p(33), p(263)]; // outer eye corners
  if (Math.hypot(r[0] - l[0], r[1] - l[1]) < 48) issues.push('small');
  // Yaw: how far the nose tip sits from the middle of the cheeks, along the cheek-to-cheek line, relative to
  // half the face width (vertical offset doesn't count: the nose is always below that line).
  const [a, b, nose] = [p(234), p(454), p(1)];
  const ax = [b[0] - a[0], b[1] - a[1]], len = Math.hypot(ax[0], ax[1]);
  const along = len ? ((nose[0] - (a[0] + b[0]) / 2) * ax[0] + (nose[1] - (a[1] + b[1]) / 2) * ax[1]) / len : 0;
  if (len && Math.abs(along) / (len / 2) > 0.35) issues.push('turned');
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const mean = sum / gray.length;
  if (mean < 55) issues.push('dark');
  else if (mean > 205) issues.push('bright');
  if (sharpness(gray, size) < 40) issues.push('blurry');
  return issues;
}

/** Variance of the Laplacian over the inner part of a square grey image: low means blurry. */
export function sharpness(gray: ArrayLike<number>, size: number) {
  const vals: number[] = [];
  const m = Math.round(size * 0.2);
  for (let y = m; y < size - m; y++) {
    for (let x = m; x < size - m; x++) {
      const i = y * size + x;
      vals.push(4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - size] - gray[i + size]);
    }
  }
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  return vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
}
