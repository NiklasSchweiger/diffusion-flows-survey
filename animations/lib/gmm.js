// Exact fields for a 2D isotropic Gaussian mixture under an interpolant
//   x_t = alpha(t) x1 + sigma(t) x0,  x0 ~ N(0, I) (noise),  x1 ~ sum_k w_k N(mu_k, s_k^2 I) (data).
// t = 0 is noise, t = 1 is data (flow-matching convention, matches paper/macros.tex).
// Marginal: p_t = sum_k w_k N(alpha mu_k, v_k I), v_k = alpha^2 s_k^2 + sigma^2.

export const linearSchedule = {
  alpha: (t) => t, sigma: (t) => 1 - t,
  dalpha: () => 1, dsigma: () => -1,
};

export class GMM {
  constructor(means, std, { weights, schedule = linearSchedule } = {}) {
    this.means = means;
    this.K = means.length;
    this.std = Array.isArray(std) ? std : means.map(() => std);
    const w = weights ?? means.map(() => 1);
    const z = w.reduce((a, b) => a + b, 0);
    this.logw = w.map((x) => Math.log(x / z));
    this.cumw = [];
    let c = 0;
    for (const x of w) this.cumw.push((c += x / z));
    this.schedule = schedule;
    this._r = new Float64Array(this.K);
  }

  sample(rng) {
    const u = rng.uniform();
    let k = this.cumw.findIndex((c) => u <= c);
    if (k < 0) k = this.K - 1;
    const [mx, my] = this.means[k];
    return { x: mx + this.std[k] * rng.gauss(), y: my + this.std[k] * rng.gauss(), k };
  }

  // Posterior responsibilities gamma_k(x) at time t (written into this._r), returns nothing.
  _resp(x, y, t) {
    const a = this.schedule.alpha(t), s = this.schedule.sigma(t);
    let mx = -Infinity;
    for (let k = 0; k < this.K; k++) {
      const v = a * a * this.std[k] ** 2 + s * s;
      const dx = x - a * this.means[k][0], dy = y - a * this.means[k][1];
      const l = this.logw[k] - (dx * dx + dy * dy) / (2 * v) - Math.log(v);
      this._r[k] = l;
      if (l > mx) mx = l;
    }
    let z = 0;
    for (let k = 0; k < this.K; k++) z += (this._r[k] = Math.exp(this._r[k] - mx));
    for (let k = 0; k < this.K; k++) this._r[k] /= z;
  }

  // Returns {u: velocity, score, d1: E[x1|x_t], d0: E[x0|x_t]} at (x, y, t).
  fields(x, y, t) {
    const S = this.schedule, a = S.alpha(t), s = S.sigma(t), da = S.dalpha(t), ds = S.dsigma(t);
    this._resp(x, y, t);
    let e1x = 0, e1y = 0, e0x = 0, e0y = 0, scx = 0, scy = 0;
    for (let k = 0; k < this.K; k++) {
      const g = this._r[k];
      if (g < 1e-12) continue;
      const s2 = this.std[k] ** 2, v = a * a * s2 + s * s;
      const rx = x - a * this.means[k][0], ry = y - a * this.means[k][1];
      e1x += g * (this.means[k][0] + (a * s2 / v) * rx);
      e1y += g * (this.means[k][1] + (a * s2 / v) * ry);
      e0x += g * (s / v) * rx;
      e0y += g * (s / v) * ry;
      scx -= g * rx / v;
      scy -= g * ry / v;
    }
    return {
      u: [da * e1x + ds * e0x, da * e1y + ds * e0y],
      score: [scx, scy],
      d1: [e1x, e1y],
      d0: [e0x, e0y],
    };
  }

  velocity(x, y, t) { return this.fields(x, y, t).u; }

  // Index of the most responsible component at time t (used for "destination" colouring).
  argmaxComponent(x, y, t = 1) {
    this._resp(x, y, t);
    let best = 0;
    for (let k = 1; k < this.K; k++) if (this._r[k] > this._r[best]) best = k;
    return best;
  }

  // Heun integration of dx/dt = u_t(x) on [t0, t1]. Returns Float32Array of (steps+1) xy pairs.
  integrateODE(x, y, steps, t0 = 0, t1 = 1) {
    const out = new Float32Array((steps + 1) * 2);
    const h = (t1 - t0) / steps;
    out[0] = x; out[1] = y;
    for (let i = 0; i < steps; i++) {
      const t = t0 + i * h;
      const [ax, ay] = this.velocity(x, y, t);
      const [bx, by] = this.velocity(x + h * ax, y + h * ay, t + h);
      x += 0.5 * h * (ax + bx);
      y += 0.5 * h * (ay + by);
      out[2 * i + 2] = x; out[2 * i + 3] = y;
    }
    return out;
  }

  // Euler-Maruyama for dx = [u + g^2/2 * score] dt + g dW. Same marginals p_t as the ODE for any g.
  integrateSDE(x, y, steps, g, rng, t0 = 0, t1 = 1) {
    const out = new Float32Array((steps + 1) * 2);
    const h = (t1 - t0) / steps, sq = Math.sqrt(h);
    out[0] = x; out[1] = y;
    for (let i = 0; i < steps; i++) {
      const t = t0 + i * h, gt = g(t);
      const f = this.fields(x, y, t);
      x += h * (f.u[0] + 0.5 * gt * gt * f.score[0]) + gt * sq * rng.gauss();
      y += h * (f.u[1] + 0.5 * gt * gt * f.score[1]) + gt * sq * rng.gauss();
      out[2 * i + 2] = x; out[2 * i + 3] = y;
    }
    return out;
  }
}

// Position on a stored trajectory at fractional time tau in [0, 1].
export function trajAt(traj, tau) {
  const steps = traj.length / 2 - 1;
  const f = Math.min(Math.max(tau, 0), 1) * steps;
  const i = Math.min(Math.floor(f), steps - 1), w = f - i;
  return [traj[2 * i] * (1 - w) + traj[2 * i + 2] * w, traj[2 * i + 1] * (1 - w) + traj[2 * i + 3] * w];
}
