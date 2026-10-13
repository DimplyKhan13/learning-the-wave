/**
 * Diagnostics: the same short, seeded training run, on any backend.
 *
 * Run it in Node (scripts/diagnose.mts) to produce reference numbers, and in a
 * browser (?diagnose=1) to compare against them. With a fixed seed the error
 * after each step should match closely; where it stops matching is where a
 * backend computes something differently.
 */
import * as tf from '@tensorflow/tfjs';
import { PaintingNet, type ModelConfig } from './model';

export interface Report {
  backend: string;
  /** Mean of each color channel of the target, to check the image decoded identically. */
  targetMean: number[];
  /** Mean of the encoded input, to check the network receives identical numbers. */
  inputMean: number;
  /** Error (MSE) before training (step 0) and after selected steps. */
  losses: Record<number, number>;
  /** Milliseconds per step when waiting for the result after every step. */
  msPerStepSynced: number;
  /** Milliseconds per step when queueing 50 steps and waiting once. */
  msPerStepQueued: number;
}

export const CHECK_STEPS = [1, 2, 5, 10, 20, 50, 100];

export async function diagnose(pixels: tf.Tensor3D, config: ModelConfig): Promise<Report> {
  const net = new PaintingNet(pixels, config);
  const read = async (t: tf.Tensor) => {
    const v = Array.from(await t.data());
    t.dispose();
    return v;
  };
  const targetMean = await read(tf.tidy(() => pixels.mean([0, 1])));
  const inputMean = await net.inputMean();

  const losses: Record<number, number> = {};
  const { colors, accuracy } = net.evaluate();
  losses[0] = (await read(tf.losses.meanSquaredError(pixels.reshape([-1, 3]), colors)))[0];
  tf.dispose([colors, accuracy]);

  let started = performance.now();
  for (let step = 1; step <= CHECK_STEPS[CHECK_STEPS.length - 1]; step++) {
    const loss = net.step(true)!;
    const value = (await read(loss))[0];
    if (CHECK_STEPS.includes(step)) losses[step] = value;
  }
  const msPerStepSynced = (performance.now() - started) / CHECK_STEPS[CHECK_STEPS.length - 1];

  started = performance.now();
  for (let i = 0; i < 49; i++) net.step();
  await read(net.step(true)!);
  const msPerStepQueued = (performance.now() - started) / 50;

  net.dispose();
  return { backend: tf.getBackend(), targetMean, inputMean, losses, msPerStepSynced, msPerStepQueued };
}
