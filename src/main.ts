/**
 * Page wiring: pick a backend, load the painting, connect the buttons.
 */
import * as tf from '@tensorflow/tfjs';
import { DEFAULT_CONFIG, PIXEL_TOLERANCE, psnr, type ModelConfig } from './model';
import { MetricChart } from './chart';
import { drawTiles, formulaHtml, inputTiles } from './equation';
import { Trainer } from './trainer';
import { diagnose, type Report } from './diagnose';
import reference from './diagnose-reference.json';
// Bundle the WASM fallback with the site instead of fetching it from a CDN.
import wasmPlain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import wasmSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreaded from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const original = $<HTMLImageElement>('original');
const painting = $<HTMLCanvasElement>('painting');
const stepBadge = $('step-badge');
const paintingCaption = $('painting-caption');
const eqCount = $('eq-count');
const milestonesEl = $('milestones');

/** Accuracy levels worth writing down, like SLO thresholds on a dashboard. */
const MILESTONES = [0.5, 0.9, 0.95, 0.99, 0.999];
const status = $('status');
const playButton = $<HTMLButtonElement>('play');
const restartButton = $<HTMLButtonElement>('restart');
const toggleButtons = [...document.querySelectorAll<HTMLButtonElement>('.toggle button')];

const IMAGE_URL = `${import.meta.env.BASE_URL}great-wave.png`;

const params = new URLSearchParams(location.search);

/**
 * WebGPU first: on a real GPU (RTX 3050) WebGL trained measurably worse, with
 * 5× the reference error after 100 steps, while WebGPU matched the reference
 * within 0.2% and ran 3× faster. WebGL, WASM and plain JS are the fallbacks.
 * `?backend=webgl|wasm|cpu` forces one, for debugging.
 */
async function pickBackend(): Promise<string> {
  const forced = params.get('backend');
  const order = forced ? [forced] : ['webgpu', 'webgl', 'wasm', 'cpu'];
  for (const name of order) {
    try {
      if (name === 'webgpu') await import('@tensorflow/tfjs-backend-webgpu');
      if (name === 'wasm') {
        const { setWasmPaths } = await import('@tensorflow/tfjs-backend-wasm');
        // Single-threaded: threads need cross-origin isolation, which static hosts can't provide.
        tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', false);
        setWasmPaths({
          'tfjs-backend-wasm.wasm': wasmPlain,
          'tfjs-backend-wasm-simd.wasm': wasmSimd,
          'tfjs-backend-wasm-threaded-simd.wasm': wasmThreaded, // required by the map, never loaded
        });
      }
      if (await tf.setBackend(name)) return name;
    } catch (err) {
      console.warn(`backend ${name} unavailable`, err); // try the next one
    }
  }
  throw new Error('no TensorFlow.js backend available');
}

/** Which GPU is doing the work (and, on WebGL, at what precision), to diagnose slow or odd runs. */
async function gpuInfo(backend: string): Promise<string> {
  if (backend === 'webgpu') {
    type Adapter = { info?: { vendor?: string; architecture?: string; description?: string } };
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<Adapter | null> } }).gpu;
    const info = (await gpu?.requestAdapter())?.info;
    return info ? info.description || [info.vendor, info.architecture].filter(Boolean).join(' ') : '';
  }
  const gl = (tf.backend() as unknown as { gpgpu?: { gl: WebGLRenderingContext } }).gpgpu?.gl;
  if (!gl) return '';
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const name = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  // "ANGLE (Vendor, Model Direct3D11 ...)" -> "Model"
  const model = name.match(/^ANGLE \([^,]+, ([^,]+?)(?: Direct3D| OpenGL| Metal|,)/)?.[1] ?? name;
  const precision = tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') ? 'float32' : 'float16';
  return `${model.slice(0, 48)}, ${precision}`;
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  original.src = url;
  await original.decode();
  return original;
}

async function main(): Promise<void> {
  const backend = await pickBackend();
  const device = await gpuInfo(backend);
  const image = await loadImage(IMAGE_URL);
  painting.width = image.naturalWidth;
  painting.height = image.naturalHeight;
  document.documentElement.style.setProperty('--image-aspect', `${image.naturalWidth} / ${image.naturalHeight}`);

  // The target: every pixel's color as a number between 0 and 1.
  const pixels = tf.tidy(() => tf.browser.fromPixels(image).toFloat().div(255) as tf.Tensor3D);

  // Without a GPU the full-size network is too slow to be fun, so shrink it.
  const base: ModelConfig =
    backend === 'webgpu' || backend === 'webgl' ? DEFAULT_CONFIG : { ...DEFAULT_CONFIG, frequencies: 64, hiddenLayers: 2, hiddenUnits: 64 };
  let config = base;

  // ?diagnose=1: run the seeded check from scripts/diagnose.mts here and compare.
  if (params.has('diagnose')) {
    status.textContent = `Running diagnostics on ${backend}${device ? ` (${device})` : ''}…`;
    await runDiagnostics(image, device);
    status.textContent = 'Diagnostics done. Reload without ?diagnose=1 to train.';
    return;
  }

  // Main chart: share of pixels painted right. The log-scale error chart is
  // kept, hidden; `?chart=error` swaps it back in.
  const showError = params.get('chart') === 'error';
  $('accuracy-panel').hidden = showError;
  $('error-panel').hidden = !showError;
  const accuracyChart = new MetricChart($<HTMLCanvasElement>('accuracy-chart'), {
    title: `PIXELS PAINTED RIGHT (±${PIXEL_TOLERANCE * 100}%)`,
    scale: 'percent',
  });
  const errorChart = new MetricChart($<HTMLCanvasElement>('chart'), { title: 'ERROR (MSE, log scale)', scale: 'log' });
  let lastLoss = NaN;
  let reached: { step: number; seconds: number }[] = [];
  const aspect = image.naturalWidth / image.naturalHeight;

  const trainer = new Trainer(pixels, {
    onLoss(step, loss) {
      lastLoss = loss;
      errorChart.push(step, loss);
    },
    async onPaint(step, img, accuracy) {
      await tf.browser.toPixels(img, painting);
      accuracyChart.push(step, accuracy);
      // Log each threshold the first time it's crossed, with steps and wall time on this device.
      while (reached.length < MILESTONES.length && accuracy >= MILESTONES[reached.length]) {
        reached.push({ step, seconds: trainer.elapsedSeconds });
      }
      milestonesEl.textContent = reached
        .map((m, i) => `${MILESTONES[i] * 100}% right: step ${m.step.toLocaleString('en-US')}, ${m.seconds.toFixed(1)} s`)
        .join(' · ');
      stepBadge.textContent = `step ${step.toLocaleString('en-US')}`;
    },
    onStatus({ running, step, stepsPerSecond }) {
      playButton.textContent = running ? 'Pause' : step === 0 ? 'Play' : 'Resume';
      const parts = [
        device ? `${backend.toUpperCase()} (${device})` : backend.toUpperCase(),
        stepsPerSecond ? `${stepsPerSecond.toFixed(0)} steps/s` : '',
        `step ${step.toLocaleString('en-US')}`,
        Number.isFinite(lastLoss) ? `PSNR ${psnr(lastLoss).toFixed(1)} dB` : '',
      ];
      status.textContent = parts.filter(Boolean).join(' · ');
      const numbers = trainer.net?.parameterCount.toLocaleString('en-US') ?? '';
      eqCount.innerHTML = `<strong>${numbers}</strong> numbers (the W's and b's), adjusted <strong>${step.toLocaleString('en-US')}</strong> times so far.`;
    },
  });

  const label = (c: ModelConfig) => (c.fourier ? 'with Fourier features' : 'without Fourier features');
  const start = async (next: ModelConfig) => {
    config = next;
    lastLoss = NaN;
    reached = [];
    accuracyChart.reset(label(config));
    errorChart.reset(label(config));
    toggleButtons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.fourier === String(config.fourier))));
    trainer.pause();
    await trainer.restart(config);
    await showEquation();
    trainer.play();
  };

  // The equation panel follows the current mode (with or without Fourier features).
  const showEquation = async () => {
    $('formula').innerHTML = formulaHtml(config);
    $('tiles-intro').textContent = config.fourier
      ? `What the network actually receives: (x, y) turned into ${config.frequencies * 2} striped patterns. A few of them, from slow to fast:`
      : 'What the network receives without Fourier features: just two smooth ramps, x and y. Fine detail is hard to build from these.';
    drawTiles($('tiles'), await inputTiles(trainer.net!), aspect);
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    accuracyChart.draw();
    errorChart.draw();
    void showEquation();
  });

  // Hover or touch the painting: call the function at that exact point, even between pixels.
  let probing = false;
  const probe = async (event: PointerEvent) => {
    if (probing || !trainer.net) return;
    probing = true;
    const rect = painting.getBoundingClientRect();
    const toCoord = (offset: number, size: number, pixels: number) =>
      -1 + (2 * ((offset / size) * pixels - 0.5)) / (pixels - 1);
    const x = toCoord(event.clientX - rect.left, rect.width, painting.width);
    const y = toCoord(event.clientY - rect.top, rect.height, painting.height);
    const color = trainer.net.colorAt(x, y);
    const [r, g, b] = Array.from(await color.data(), (v) => Math.round(v * 255));
    color.dispose();
    const fmt = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
    paintingCaption.innerHTML = `<span class="swatch" style="background: rgb(${r} ${g} ${b})"></span>f(${fmt(x)}, ${fmt(y)}) = rgb(${r}, ${g}, ${b})`;
    probing = false;
  };
  painting.addEventListener('pointermove', probe);
  painting.addEventListener('pointerdown', probe);
  painting.addEventListener('pointerleave', () => (paintingCaption.textContent = 'What the network paints'));

  playButton.addEventListener('click', () => (trainer.running ? trainer.pause() : trainer.play()));
  restartButton.addEventListener('click', () => void start(config));
  for (const button of toggleButtons) {
    button.addEventListener('click', () => {
      const fourier = button.dataset.fourier === 'true';
      if (fourier !== config.fourier) void start({ ...base, fourier });
    });
  }

  // Same check as ?diagnose=1, from a button (the URL may not reach an embedded page).
  const diagnoseButton = $<HTMLButtonElement>('diagnose');
  diagnoseButton.addEventListener('click', async () => {
    trainer.pause();
    diagnoseButton.disabled = true;
    diagnoseButton.textContent = 'Running diagnostics…';
    await runDiagnostics(image, device);
    diagnoseButton.textContent = 'Reload the page to train again';
  });

  playButton.disabled = restartButton.disabled = false;
  await start(config);

  // Handy for debugging from the console.
  Object.assign(window, { wave: { tf, trainer, accuracyChart, errorChart } });
}

/**
 * Run the seeded check on each GPU path this browser offers and compare every
 * one against the Node reference. Leaves TensorFlow.js on the last backend
 * tried, so the page asks for a reload afterwards.
 */
async function runDiagnostics(image: HTMLImageElement, device: string): Promise<void> {
  const variants: { label: string; backend: string }[] = [
    { label: 'WebGPU', backend: 'webgpu' },
    { label: 'WebGL', backend: 'webgl' },
  ];
  const results: { label: string; report?: Report; error?: string }[] = [];
  for (const variant of variants) {
    status.textContent = `Diagnostics: ${variant.label}…`;
    try {
      if (variant.backend === 'webgpu') await import('@tensorflow/tfjs-backend-webgpu');
      if (!(await tf.setBackend(variant.backend))) throw new Error('not available');
      const pixels = tf.tidy(() => tf.browser.fromPixels(image).toFloat().div(255) as tf.Tensor3D);
      results.push({ label: variant.label, report: await diagnose(pixels, DEFAULT_CONFIG) });
      pixels.dispose();
    } catch (err) {
      results.push({ label: variant.label, error: err instanceof Error ? err.message : String(err) });
    }
  }
  showDiagnostics(results, device);
}

function showDiagnostics(results: { label: string; report?: Report; error?: string }[], device: string): void {
  const ref = reference as Report;
  const fmt = (v: number) => v.toPrecision(4);
  // A cell per variant, red when more than 5% away from the reference.
  const cell = (v: number | undefined, r: number, compare = true) => {
    if (v === undefined) return '<td>—</td>';
    const pct = (Math.abs(v - r) / Math.max(Math.abs(r), 1e-9)) * 100;
    return compare ? `<td class="${pct > 5 ? 'bad' : 'ok'}">${fmt(v)} <small>(${pct.toFixed(1)}%)</small></td>` : `<td>${fmt(v)}</td>`;
  };
  const row = (label: string, pick: (r: Report) => number, compare = true) =>
    `<tr><td>${label}</td>${results.map((x) => cell(x.report && pick(x.report), pick(ref), compare)).join('')}<td>${fmt(pick(ref))}</td></tr>`;
  const rows = [
    ...[0, 1, 2].map((i) => row(`target mean ${'RGB'[i]}`, (r) => r.targetMean[i])),
    row('input mean', (r) => r.inputMean),
    ...Object.keys(ref.losses).map((step) => row(`error after step ${step}`, (r) => r.losses[Number(step)])),
    row('ms/step, waiting each step', (r) => r.msPerStepSynced, false),
    row('ms/step, 50 queued', (r) => r.msPerStepQueued, false),
  ];
  const errors = results.filter((x) => x.error).map((x) => `<p>${x.label}: ${x.error}</p>`).join('');
  document.querySelector('.diagnose')?.remove();
  const section = document.createElement('section');
  section.className = 'diagnose';
  section.innerHTML = `<h2>Diagnostics</h2>
    <p>${device || 'no GPU info'} · ${navigator.userAgent}</p>
    <table><thead><tr><th></th>${results.map((x) => `<th>${x.label}</th>`).join('')}<th>reference (tfjs-node, CPU)</th></tr></thead>
    <tbody>${rows.join('')}</tbody></table>${errors}`;
  document.querySelector('.demo')!.after(section);
}

main().catch((err) => {
  console.error(err);
  status.textContent = `Something went wrong: ${err instanceof Error ? err.message : String(err)}`;
});
