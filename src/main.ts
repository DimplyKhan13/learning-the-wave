/**
 * Page wiring: pick a backend, load the painting, connect the buttons.
 */
import * as tf from '@tensorflow/tfjs';
import { DEFAULT_CONFIG, psnr, type ModelConfig } from './model';
import { LossChart } from './chart';
import { Trainer } from './trainer';
// Bundle the WASM fallback with the site instead of fetching it from a CDN.
import wasmPlain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import wasmSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreaded from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const original = $<HTMLImageElement>('original');
const painting = $<HTMLCanvasElement>('painting');
const stepBadge = $('step-badge');
const status = $('status');
const playButton = $<HTMLButtonElement>('play');
const restartButton = $<HTMLButtonElement>('restart');
const toggleButtons = [...document.querySelectorAll<HTMLButtonElement>('.toggle button')];

const IMAGE_URL = `${import.meta.env.BASE_URL}great-wave.png`;

const params = new URLSearchParams(location.search);

/**
 * WebGL runs on the GPU; WASM and plain JS are the fallbacks.
 * `?backend=wasm` (or cpu) forces one, for debugging and for recording the video.
 */
async function pickBackend(): Promise<string> {
  const forced = params.get('backend');
  const order = forced ? [forced] : ['webgl', 'wasm', 'cpu'];
  for (const name of order) {
    try {
      if (name === 'wasm') {
        const { setWasmPaths } = await import('@tensorflow/tfjs-backend-wasm');
        setWasmPaths({
          'tfjs-backend-wasm.wasm': wasmPlain,
          'tfjs-backend-wasm-simd.wasm': wasmSimd,
          'tfjs-backend-wasm-threaded-simd.wasm': wasmThreaded,
        });
      }
      if (await tf.setBackend(name)) return name;
    } catch {
      // try the next one
    }
  }
  throw new Error('no TensorFlow.js backend available');
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  original.src = url;
  await original.decode();
  return original;
}

async function main(): Promise<void> {
  const backend = await pickBackend();
  const image = await loadImage(IMAGE_URL);
  painting.width = image.naturalWidth;
  painting.height = image.naturalHeight;
  document.documentElement.style.setProperty('--image-aspect', `${image.naturalWidth} / ${image.naturalHeight}`);

  // The target: every pixel's color as a number between 0 and 1.
  const pixels = tf.tidy(() => tf.browser.fromPixels(image).toFloat().div(255) as tf.Tensor3D);

  // Without a GPU the full-size network is too slow to be fun, so shrink it.
  const base: ModelConfig =
    backend === 'webgl' ? DEFAULT_CONFIG : { ...DEFAULT_CONFIG, frequencies: 64, hiddenLayers: 2, hiddenUnits: 64 };
  let config = base;

  const chart = new LossChart($<HTMLCanvasElement>('chart'));
  let lastLoss = NaN;
  let chartQueued = false;

  const trainer = new Trainer(pixels, {
    onLoss(step, loss) {
      lastLoss = loss;
      chart.push(step, loss);
      // Redraw the chart at most once per frame.
      if (!chartQueued) {
        chartQueued = true;
        requestAnimationFrame(() => {
          chartQueued = false;
          chart.draw();
        });
      }
    },
    async onPaint(step, img) {
      await tf.browser.toPixels(img, painting);
      stepBadge.textContent = `step ${step.toLocaleString('en-US')}`;
    },
    onStatus({ running, step, stepsPerSecond }) {
      playButton.textContent = running ? 'Pause' : step === 0 ? 'Play' : 'Resume';
      const parts = [
        backend.toUpperCase(),
        stepsPerSecond ? `${stepsPerSecond.toFixed(0)} steps/s` : '',
        `step ${step.toLocaleString('en-US')}`,
        Number.isFinite(lastLoss) ? `PSNR ${psnr(lastLoss).toFixed(1)} dB` : '',
      ];
      status.textContent = parts.filter(Boolean).join(' · ');
    },
  });

  const label = (c: ModelConfig) => (c.fourier ? 'with Fourier features' : 'without Fourier features');
  const start = async (next: ModelConfig) => {
    config = next;
    lastLoss = NaN;
    chart.reset(label(config));
    toggleButtons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.fourier === String(config.fourier))));
    trainer.pause();
    await trainer.restart(config);
    trainer.play();
  };

  playButton.addEventListener('click', () => (trainer.running ? trainer.pause() : trainer.play()));
  restartButton.addEventListener('click', () => void start(config));
  for (const button of toggleButtons) {
    button.addEventListener('click', () => {
      const fourier = button.dataset.fourier === 'true';
      if (fourier !== config.fourier) void start({ ...base, fourier });
    });
  }

  playButton.disabled = restartButton.disabled = false;
  await start(config);

  // Handy for debugging from the console.
  Object.assign(window, { wave: { tf, trainer, chart } });
}

main().catch((err) => {
  console.error(err);
  status.textContent = `Something went wrong: ${err instanceof Error ? err.message : String(err)}`;
});
