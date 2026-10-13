/**
 * The whole "AI" of this project.
 *
 * A small neural network that never sees the picture. It gets a pixel's
 * coordinate (x, y) and has to answer one question: what color goes here?
 * Training = guess every pixel, measure how wrong the guesses are, nudge the
 * weights to be a little less wrong, repeat.
 */
import * as tf from '@tensorflow/tfjs';

export interface ModelConfig {
  /** Encode (x, y) with Fourier features before the MLP (Tancik et al., 2020). */
  fourier: boolean;
  /** Number of random frequencies in B. The encoding has 2× this many values. */
  frequencies: number;
  /** Standard deviation of B: higher = the network can draw finer detail. */
  sigma: number;
  hiddenLayers: number;
  hiddenUnits: number;
  learningRate: number;
  /** Fixed seed so every visitor watches the same run. */
  seed: number;
}

export const DEFAULT_CONFIG: ModelConfig = {
  fourier: true,
  frequencies: 128,
  sigma: 8,
  hiddenLayers: 3,
  hiddenUnits: 128,
  learningRate: 1e-3,
  seed: 1831, // the year Hokusai published the print
};

/** One (x, y) per pixel, row by row, both normalized to [-1, 1]. Shape [H*W, 2]. */
export function pixelCoordinates(width: number, height: number): tf.Tensor2D {
  return tf.tidy(() => {
    const xs = tf.linspace(-1, 1, width);
    const ys = tf.linspace(-1, 1, height);
    const x = tf.tile(xs.reshape([1, width]), [height, 1]);
    const y = tf.tile(ys.reshape([height, 1]), [1, width]);
    return tf.stack([x.flatten(), y.flatten()], 1) as tf.Tensor2D;
  });
}

/**
 * γ(v) = [sin(2πBv), cos(2πBv)].
 *
 * A plain MLP is biased towards smooth functions ("spectral bias"), so given
 * raw (x, y) it settles for a blur. Projecting the coordinates onto many random
 * sine waves hands it high frequencies for free, and detail becomes learnable.
 */
export function fourierFeatures(coords: tf.Tensor2D, B: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const projected = coords.matMul(B).mul(2 * Math.PI);
    return tf.concat([projected.sin(), projected.cos()], 1) as tf.Tensor2D;
  });
}

export class PaintingNet {
  readonly config: ModelConfig;
  private readonly inputs: tf.Tensor2D; // encoded coordinates, computed once
  private readonly target: tf.Tensor2D; // the real colors, in [0, 1]
  private readonly params: tf.Variable[] = [];
  private readonly optimizer: tf.Optimizer;

  /** @param pixels the original image, shape [H, W, 3], values in [0, 1]. */
  constructor(pixels: tf.Tensor3D, config: ModelConfig) {
    this.config = config;
    const [height, width] = pixels.shape;
    this.target = pixels.reshape([height * width, 3]) as tf.Tensor2D;

    // The input never changes, so encode it once instead of on every step.
    this.inputs = tf.tidy(() => {
      const coords = pixelCoordinates(width, height);
      if (!config.fourier) return coords;
      const B = tf.randomNormal<tf.Rank.R2>([2, config.frequencies], 0, config.sigma, 'float32', config.seed);
      return fourierFeatures(coords, B);
    });

    // A stack of dense layers: ReLU in the middle, 3 outputs (r, g, b) at the end.
    const sizes = [this.inputs.shape[1], ...Array(config.hiddenLayers).fill(config.hiddenUnits), 3];
    for (let i = 0; i < sizes.length - 1; i++) {
      const limit = Math.sqrt(6 / sizes[i]); // He-uniform init, good for ReLU
      const w = tf.randomUniform([sizes[i], sizes[i + 1]], -limit, limit, 'float32', config.seed + i + 1);
      this.params.push(tf.variable(w), tf.variable(tf.zeros([sizes[i + 1]])));
      w.dispose();
    }
    this.optimizer = tf.train.adam(config.learningRate);
  }

  /** The network's painting: one color per pixel, shape [H*W, 3], values in [0, 1]. */
  predict(): tf.Tensor2D {
    return tf.tidy(() => {
      let h: tf.Tensor2D = this.inputs;
      for (let i = 0; i < this.params.length; i += 2) {
        h = h.matMul(this.params[i] as tf.Tensor2D).add(this.params[i + 1]);
        h = i + 2 < this.params.length ? h.relu() : h.sigmoid();
      }
      return h;
    });
  }

  /**
   * One training step: paint everything, measure the error (mean squared
   * difference from the original), and let Adam nudge every weight downhill.
   * Returns the error only when asked, because reading it back from the GPU
   * forces a sync and slows the loop.
   */
  step(returnLoss = false): tf.Scalar | null {
    const loss = () => tf.losses.meanSquaredError(this.target, this.predict()) as tf.Scalar;
    return this.optimizer.minimize(loss, returnLoss, this.params);
  }

  dispose(): void {
    tf.dispose([this.inputs, this.target, ...this.params]);
    this.optimizer.dispose();
  }
}

/** Peak signal-to-noise ratio in dB, the usual "how close is the image" score. */
export const psnr = (mse: number): number => -10 * Math.log10(mse);
