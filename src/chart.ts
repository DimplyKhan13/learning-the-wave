/**
 * The error chart, drawn by hand on a <canvas>: no chart library, just lines.
 * Styled like a monitoring panel: log-scale y axis, grid, current value.
 */

export interface Series {
  label: string;
  steps: number[];
  losses: number[];
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class LossChart {
  private readonly ctx: CanvasRenderingContext2D;
  /** The run being trained right now. */
  current: Series = { label: '', steps: [], losses: [] };
  /** The previous run, kept faded in the background for comparison. */
  baseline: Series | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  reset(label: string): void {
    if (this.current.steps.length > 1) this.baseline = this.current;
    this.current = { label, steps: [], losses: [] };
    this.draw();
  }

  push(step: number, loss: number): void {
    this.current.steps.push(step);
    this.current.losses.push(loss);
  }

  draw(): void {
    const { canvas, ctx } = this;
    // Render at the device's pixel density so lines stay sharp on phones.
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const font = css('--font-mono') || 'monospace';
    const pad = { left: 44, right: 12, top: 30, bottom: 22 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const runs = [this.baseline, this.current].filter((s): s is Series => !!s && s.steps.length > 0);
    const all = runs.flatMap((s) => s.losses).filter((l) => l > 0);

    // Axis ranges: x is the step count, y is log10(error) snapped to whole decades.
    const maxStep = Math.max(100, ...runs.map((s) => s.steps[s.steps.length - 1]));
    const yMax = all.length ? Math.ceil(Math.log10(Math.max(...all))) : 0;
    const yMin = all.length ? Math.min(yMax - 1, Math.floor(Math.log10(Math.min(...all)))) : -3;
    const x = (step: number) => pad.left + (step / maxStep) * plotW;
    const y = (loss: number) => pad.top + ((yMax - Math.log10(loss)) / (yMax - yMin)) * plotH;

    // Grid: one line per decade, plus a few vertical step markers.
    ctx.font = `11px ${font}`;
    ctx.lineWidth = 1;
    ctx.fillStyle = css('--muted');
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (let d = yMin; d <= yMax; d++) {
      const gy = Math.round(y(10 ** d)) + 0.5;
      ctx.strokeStyle = css('--grid');
      ctx.beginPath();
      ctx.moveTo(pad.left, gy);
      ctx.lineTo(w - pad.right, gy);
      ctx.stroke();
      ctx.fillText(`1e${d}`, pad.left - 6, gy);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const tick = niceTick(maxStep / 4);
    for (let s = 0; s <= maxStep; s += tick) ctx.fillText(formatSteps(s), x(s), h - pad.bottom + 6);

    // Curves: the faded baseline first, the live run on top.
    for (const series of runs) {
      const live = series === this.current;
      ctx.strokeStyle = live ? css('--accent') : css('--baseline');
      ctx.lineWidth = live ? 2 : 1.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      series.steps.forEach((s, i) => {
        const px = x(s);
        const py = y(Math.max(series.losses[i], 10 ** yMin));
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // Header: panel title on the left, live readout on the right.
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillStyle = css('--muted');
    ctx.font = `600 11px ${font}`;
    ctx.fillText('ERROR (MSE, log scale)', pad.left, 16);
    const n = this.current.losses.length;
    if (n) {
      ctx.textAlign = 'right';
      ctx.fillStyle = css('--accent');
      ctx.font = `600 13px ${font}`;
      ctx.fillText(this.current.losses[n - 1].toExponential(2), w - pad.right, 16);
    }
    if (this.baseline) {
      ctx.textAlign = 'left';
      ctx.fillStyle = css('--baseline');
      ctx.font = `11px ${font}`;
      ctx.fillText(`— previous: ${this.baseline.label}`, pad.left + 4, h - pad.bottom - 8);
    }
  }
}

/** Round a raw tick spacing up to 1, 2 or 5 × a power of ten. */
function niceTick(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((t) => t >= raw)!;
}

const formatSteps = (s: number) => (s >= 1000 ? `${s / 1000}k` : `${s}`);
