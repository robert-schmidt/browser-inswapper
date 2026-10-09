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
