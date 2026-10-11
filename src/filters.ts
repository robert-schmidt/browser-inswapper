// One-Euro filter (Casiez et al., 2012): smooths landmark jitter when the face is still, follows quickly when
// it moves. One filter per coordinate.

class LowPass {
  private y: number | null = null;
  filter(x: number, alpha: number) { this.y = this.y === null ? x : alpha * x + (1 - alpha) * this.y; return this.y; }
  last() { return this.y; }
  reset() { this.y = null; }
}

const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));

export class OneEuro {
  private x = new LowPass();
  private dx = new LowPass();
  private t: number | null = null;
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  constructor(minCutoff = 1.2, beta = 0.02, dCutoff = 1) { this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff; }
  filter(value: number, timeMs: number) {
    const dt = this.t === null ? 1 / 30 : Math.max(1e-3, (timeMs - this.t) / 1000);
    this.t = timeMs;
    const prev = this.x.last();
    const d = prev === null ? 0 : (value - prev) / dt;
    const ed = this.dx.filter(d, alpha(this.dCutoff, dt));
    return this.x.filter(value, alpha(this.minCutoff + this.beta * Math.abs(ed), dt));
  }
  reset() { this.x.reset(); this.dx.reset(); this.t = null; }
}

/** Smooths a fixed-length vector of coordinates (e.g. 5 key points × 2). */
export class VectorFilter {
  private filters: OneEuro[] = [];
  private minCutoff: number;
  private beta: number;
  constructor(minCutoff = 1.2, beta = 0.02) { this.minCutoff = minCutoff; this.beta = beta; }
  filter(values: number[], timeMs: number) {
    while (this.filters.length < values.length) this.filters.push(new OneEuro(this.minCutoff, this.beta));
    return values.map((v, i) => this.filters[i].filter(v, timeMs));
  }
  reset() { this.filters = []; }
}

/**
 * Smooths a face's landmarks (x/y pairs, pixels) without trailing a moving head. Per-point smoothing (VectorFilter)
 * only notices a move after ~200 ms and lags 15–20 px behind a fast one, so the swapped face slides off the real
 * one. Here the head's pose (centre, size and tilt, fitted to all points, so it's steady by itself) follows
 * quickly, and only the face's shape in the head's own frame (expressions, jitter) is smoothed hard. Simulated at
 * 30 fps: ~2 px behind at any speed (was 7–22 px), about the same stillness.
 */
export class FaceFilter {
  private pose = [0, 1, 2, 3].map(() => new OneEuro(1.5, 0.2, 5));
  private shape: VectorFilter;
  private ref: number[] | null = null;
  private radius = 1;
  constructor(minCutoff = 3.0, beta = 0.02) { this.shape = new VectorFilter(minCutoff, beta); }
  filter(v: ArrayLike<number>, timeMs: number): number[] {
    const n = v.length / 2;
    let x = 0, y = 0;
    for (let i = 0; i < v.length; i += 2) { x += v[i]; y += v[i + 1]; }
    x /= n; y /= n;
    const c = Array.from(v, (p, i) => p - (i % 2 ? y : x));
    if (!this.ref) {
      this.ref = c;
      this.radius = Math.sqrt(c.reduce((s, p) => s + p * p, 0) / n) || 1;
    }
    // The similarity taking the first shape onto this one: c ≈ scale·R(angle)·ref.
    const ref = this.ref;
    let dot = 0, cross = 0, den = 0;
    for (let i = 0; i < c.length; i += 2) { dot += ref[i] * c[i] + ref[i + 1] * c[i + 1]; cross += ref[i] * c[i + 1] - ref[i + 1] * c[i]; den += ref[i] ** 2 + ref[i + 1] ** 2; }
    const scale = Math.hypot(dot, cross) / den || 1, angle = Math.atan2(cross, dot);
    const co = Math.cos(-angle) / scale, si = Math.sin(-angle) / scale, local: number[] = [];
    for (let i = 0; i < c.length; i += 2) local.push(co * c[i] - si * c[i + 1], si * c[i] + co * c[i + 1]);
    // Size and tilt in pixels at the face's edge (×radius), so one speed setting suits all four.
    const r = this.radius, [px, py, ps, pa] = this.pose;
    const fx = px.filter(x, timeMs), fy = py.filter(y, timeMs);
    const fs = Math.exp(ps.filter(Math.log(scale) * r, timeMs) / r), fa = pa.filter(angle * r, timeMs) / r;
    const L = this.shape.filter(local, timeMs), C = Math.cos(fa) * fs, S = Math.sin(fa) * fs, out: number[] = [];
    for (let i = 0; i < L.length; i += 2) out.push(fx + C * L[i] - S * L[i + 1], fy + S * L[i] + C * L[i + 1]);
    return out;
  }
  reset() { this.pose.forEach((f) => f.reset()); this.shape.reset(); this.ref = null; }
}
