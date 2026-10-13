/**
 * "The equation" panel: shows that the whole network is one function,
 * color = f(x, y), and what the Fourier features hand it as input.
 */
import type { ModelConfig, PaintingNet } from './model';

const sub = (n: number) => `<sub>${n}</sub>`;

/** The network written out as nested functions, matching the configured depth. */
export function formulaHtml(config: ModelConfig): string {
  const layers = config.hiddenLayers + 1;
  let inner = config.fourier ? '<span class="eq-gamma">γ(x, y)</span>' : '(x, y)';
  for (let i = 1; i <= layers; i++) {
    const act = i === layers ? 'sigmoid' : 'relu';
    inner = `${act}(W${sub(i)} · ${inner} + b${sub(i)})`;
  }
  const lines = [`<span class="eq-out">color(x, y)</span> = ${inner}`];
  if (config.fourier) {
    lines.push(
      `<span class="eq-gamma">γ(x, y)</span> = [ sin(2π·B·(x, y)), cos(2π·B·(x, y)) ]` +
        `<span class="eq-note">B: ${config.frequencies} random frequencies, picked once and never trained</span>`,
    );
  }
  return lines.map((l) => `<div class="eq-line">${l}</div>`).join('');
}

export interface Tile {
  caption: string;
  /** Value at a coordinate (x, y) in [-1, 1]², in [-1, 1]. */
  fn: (x: number, y: number) => number;
}

/**
 * The inputs the network gets, as pictures. Without Fourier features that's
 * just two ramps (x and y). With them, it's dozens of striped patterns at
 * different angles and speeds; we show a handful, from slow to fast.
 */
export async function inputTiles(net: PaintingNet, count = 12): Promise<Tile[]> {
  if (!net.B) {
    return [
      { caption: 'x', fn: (x) => x },
      { caption: 'y', fn: (_, y) => y },
    ];
  }
  const data = await net.B.data();
  const m = net.B.shape[1];
  const freqs = Array.from({ length: m }, (_, k) => [data[k], data[m + k]] as const);
  freqs.sort((a, b) => Math.hypot(...a) - Math.hypot(...b));
  const fmt = (v: number) => Math.abs(v).toFixed(1);
  return Array.from({ length: count }, (_, i) => {
    const [bx, by] = freqs[Math.round((i / (count - 1)) * (m - 1) * 0.9)];
    const trig = i % 2 ? Math.cos : Math.sin;
    return {
      caption: `${i % 2 ? 'cos' : 'sin'} 2π(${fmt(bx)}x ${by < 0 ? '−' : '+'} ${fmt(by)}y)`,
      fn: (x: number, y: number) => trig(2 * Math.PI * (bx * x + by * y)),
    };
  });
}

/** Draw each tile at a higher resolution than the image so the stripes stay legible. */
export function drawTiles(container: HTMLElement, tiles: Tile[], aspect: number): void {
  const css = getComputedStyle(document.documentElement);
  const lo = hexToRgb(css.getPropertyValue('--surface'));
  const hi = hexToRgb(css.getPropertyValue('--accent'));
  const width = 192;
  const height = Math.round(width / aspect);
  container.replaceChildren(
    ...tiles.map((tile) => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d')!;
      const img = ctx.createImageData(width, height);
      for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
          const t = (tile.fn(-1 + (2 * i) / (width - 1), -1 + (2 * j) / (height - 1)) + 1) / 2;
          const o = (j * width + i) * 4;
          for (let c = 0; c < 3; c++) img.data[o + c] = lo[c] + (hi[c] - lo[c]) * t;
          img.data[o + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      const figure = document.createElement('figure');
      const caption = document.createElement('figcaption');
      caption.textContent = tile.caption;
      figure.append(canvas, caption);
      return figure;
    }),
  );
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.trim().replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
