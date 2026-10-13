/**
 * Train the exact model from the page outside the browser and log milestones:
 * how many steps and seconds until X% of pixels are painted right.
 *
 *   npx tsx scripts/benchmark.mts [maxSteps] [--no-fourier]
 *
 * Uses native TensorFlow (tfjs-node) on the CPU, so absolute times are this
 * machine's, not a browser's. Steps-to-milestone are the same everywhere.
 */
import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import { DEFAULT_CONFIG, PIXEL_TOLERANCE, PaintingNet, psnr } from '../src/model.ts';

const args = process.argv.slice(2);
const maxSteps = Number(args.find((a) => /^\d+$/.test(a)) ?? 5000);
const config = { ...DEFAULT_CONFIG, fourier: !args.includes('--no-fourier') };
const MILESTONES = [0.5, 0.9, 0.95, 0.99, 0.999];
const EVAL_EVERY = 10; // same cadence as the page's repaint

const { data, info } = await sharp('public/great-wave.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
const pixels = tf.tidy(() => tf.tensor3d(new Uint8Array(data), [info.height, info.width, 3]).toFloat().div(255) as tf.Tensor3D);
const net = new PaintingNet(pixels, config);

console.log(`${config.fourier ? 'WITH' : 'WITHOUT'} Fourier features · ${info.width}×${info.height} px · ${net.parameterCount.toLocaleString('en-US')} parameters · "right" = every channel within ±${PIXEL_TOLERANCE * 100}%`);
console.log('milestone   step    time');

const started = performance.now();
let next = 0;
let last = { step: 0, accuracy: 0, mse: 1 };
for (let step = 1; step <= maxSteps && next < MILESTONES.length; step++) {
  net.step();
  if (step % EVAL_EVERY) continue;
  const { colors, accuracy } = net.evaluate();
  const mse = tf.tidy(() => tf.losses.meanSquaredError(pixels.reshape([-1, 3]), colors).dataSync()[0]);
  last = { step, accuracy: accuracy.dataSync()[0], mse };
  tf.dispose([colors, accuracy]);
  while (next < MILESTONES.length && last.accuracy >= MILESTONES[next]) {
    const secs = (performance.now() - started) / 1000;
    console.log(`${(MILESTONES[next] * 100).toString().padStart(5)}%    ${String(step).padStart(5)}   ${secs.toFixed(1).padStart(6)} s`);
    next++;
  }
}
const secs = (performance.now() - started) / 1000;
console.log(`final: step ${last.step} · ${(last.accuracy * 100).toFixed(2)}% pixels right · MSE ${last.mse.toExponential(2)} · PSNR ${psnr(last.mse).toFixed(1)} dB · ${secs.toFixed(1)} s · ${(last.step / secs).toFixed(1)} steps/s (${tf.getBackend()}, CPU)`);
net.dispose();
