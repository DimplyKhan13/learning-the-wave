/**
 * Training charts, drawn by hand on a <canvas>: no chart library, just lines.
 * Styled like a monitoring panel: grid, axis labels, current value.
 *
 * Two flavors share this code:
 * - 'percent': a 0–100% scale (share of pixels painted right). Easy to read.
 * - 'log':     a log-scale error curve, which keeps showing progress after
 *              the percentage has saturated.
 */

export interface Series {
  label: string;
  steps: number[];
  values: number[];
}

export interface ChartOptions {
  title: string;
  scale: 'percent' | 'log';
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class MetricChart {
  private readonly ctx: CanvasRenderingContext2D;
  /** The run being trained right now. */
  current: Series = { label: '', steps: [], values: [] };
  /** The previous run, kept faded in the background for comparison. */
  baseline: Series | null = null;
  private queued = false;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly options: ChartOptions,
  ) {
    this.ctx = canvas.getContext('2d')!;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  reset(label: string): void {
    if (this.current.steps.length > 1) this.baseline = this.current;
    this.current = { label, steps: [], values: [] };
    this.draw();
  }

  push(step: number, value: number): void {
    this.current.steps.push(step);
    this.current.values.push(value);
    // Redraw at most once per frame, however often values arrive.
    if (!this.queued) {
      this.queued = true;
      requestAnimationFrame(() => {
        this.queued = false;
        this.draw();
      });
    }
  }

  draw(): void {
    const { canvas, ctx, options } = this;
    if (canvas.offsetParent === null) return; // hidden: nothing to draw
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

    // x is the step count; y is either 0–100% or log10(error) snapped to whole decades.
    const maxStep = Math.max(100, ...runs.map((s) => s.steps[s.steps.length - 1]));
    const x = (step: number) => pad.left + (step / maxStep) * plotW;
    let y: (v: number) => number;
    let gridLines: { value: number; label: string }[];
    if (options.scale === 'percent') {
      y = (v) => pad.top + (1 - v) * plotH;
      gridLines = [0, 0.25, 0.5, 0.75, 1].map((v) => ({ value: v, label: `${v * 100}%` }));
    } else {
      const all = runs.flatMap((s) => s.values).filter((v) => v > 0);
      const yMax = all.length ? Math.ceil(Math.log10(Math.max(...all))) : 0;
      const yMin = all.length ? Math.min(yMax - 1, Math.floor(Math.log10(Math.min(...all)))) : -3;
      y = (v) => pad.top + ((yMax - Math.log10(Math.max(v, 10 ** yMin))) / (yMax - yMin)) * plotH;
      gridLines = [];
      for (let d = yMin; d <= yMax; d++) gridLines.push({ value: 10 ** d, label: `1e${d}` });
    }

    // Grid and axis labels.
    ctx.font = `11px ${font}`;
    ctx.lineWidth = 1;
    ctx.fillStyle = css('--muted');
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const line of gridLines) {
      const gy = Math.round(y(line.value)) + 0.5;
      ctx.strokeStyle = css('--grid');
      ctx.beginPath();
      ctx.moveTo(pad.left, gy);
      ctx.lineTo(w - pad.right, gy);
      ctx.stroke();
      ctx.fillText(line.label, pad.left - 6, gy);
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
        if (i === 0) ctx.moveTo(x(s), y(series.values[i]));
        else ctx.lineTo(x(s), y(series.values[i]));
      });
      ctx.stroke();
    }

    // Header: panel title on the left, live readout on the right.
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillStyle = css('--muted');
    ctx.font = `600 11px ${font}`;
    ctx.fillText(options.title, pad.left, 16);
    const n = this.current.values.length;
    if (n) {
      const v = this.current.values[n - 1];
      ctx.textAlign = 'right';
      ctx.fillStyle = css('--accent');
      ctx.font = `600 13px ${font}`;
      ctx.fillText(options.scale === 'percent' ? `${(v * 100).toFixed(1)}%` : v.toExponential(2), w - pad.right, 16);
    }
    if (this.baseline) {
      // Keep the legend clear of the curve: top for the error, bottom for the percentage.
      const legendY = options.scale === 'percent' ? h - pad.bottom - 8 : pad.top + 12;
      ctx.textAlign = 'right';
      ctx.fillStyle = css('--baseline');
      ctx.font = `11px ${font}`;
      ctx.fillText(`— previous: ${this.baseline.label}`, w - pad.right - 4, legendY);
    }
  }
}

/** Round a raw tick spacing up to 1, 2 or 5 × a power of ten. */
function niceTick(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((t) => t >= raw)!;
}

const formatSteps = (s: number) => (s >= 1000 ? `${s / 1000}k` : `${s}`);
