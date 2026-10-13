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
  /** Keep the learning rate constant until this step, then shrink it exponentially... */
  decayStart: number;
  /** ...to `finalLearningRate` over this many steps (0 = constant forever). */
  decaySteps: number;
  finalLearningRate: number;
  /** Unlock the Fourier frequencies from slow to fast over this many steps (0 = all at once). */
  coarseToFineSteps: number;
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
  decayStart: 0,
  decaySteps: 0,
  finalLearningRate: 1e-3,
  coarseToFineSteps: 0,
  seed: 1831, // the year Hokusai published the print
};

/** One (x, y) per pixel, row by row, both normalized to [-1, 1]. Shape [H*W, 2]. */
export function pixelCoordinates(width: number, height: number): tf.Tensor2D {
  // Plain JavaScript, so every device starts from exactly the same numbers.
  const coords = new Float32Array(width * height * 2);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const i = 2 * (row * width + col);
      coords[i] = -1 + (2 * col) / (width - 1);
      coords[i + 1] = -1 + (2 * row) / (height - 1);
    }
  }
  return tf.tensor2d(coords, [width * height, 2]);
}

/**
 * γ(v) = [sin(2πBv), cos(2πBv)].
 *
 * A plain MLP is biased towards smooth functions ("spectral bias"), so given
 * raw (x, y) it settles for a blur. Projecting the coordinates onto many random
 * sine waves hands it high frequencies for free, and detail becomes learnable.
 */
export function fourierFeatures(coords: tf.Tensor2D, B: tf.Tensor2D): tf.Tensor2D {
  // Computed on the CPU in double precision, not on the GPU: 2πBv reaches a few
  // hundred radians, where GPU sin() is only approximate, and that quietly
  // blurs the finest detail. The input never changes, so this runs once.
  const [n, m] = [coords.shape[0], B.shape[1]];
  const v = coords.dataSync();
  const b = B.dataSync();
  const out = new Float32Array(n * 2 * m);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) {
      const angle = 2 * Math.PI * (v[2 * i] * b[k] + v[2 * i + 1] * b[m + k]);
      out[i * 2 * m + k] = Math.sin(angle);
      out[i * 2 * m + m + k] = Math.cos(angle);
    }
  }
  return tf.tensor2d(out, [n, 2 * m]);
}

/**
 * A pixel counts as "painted right" when every channel is within 1% of the
 * original (about ±2.5 levels out of 255). Strict on purpose: at 5% the curve
 * hit 99% within ~560 steps and then had nothing left to show.
 */
export const PIXEL_TOLERANCE = 0.01;

export class PaintingNet {
  readonly config: ModelConfig;
  /** The fixed random frequencies of the Fourier features, shape [2, m]; null without them. */
  readonly B: tf.Tensor2D | null;
  /** How many numbers training gets to adjust (all weights and biases). */
  readonly parameterCount: number;
  /** How many pictures this one function paints; picture i is at t = i / (frames - 1). */
  readonly frames: number;
  private readonly encoded: tf.Tensor2D; // γ(x, y) for every pixel, computed once
  private readonly inputs: tf.Tensor2D; // what training feeds in: γ(x, y), plus t when there are several pictures
  private readonly target: tf.Tensor2D; // the real colors, in [0, 1]
  private readonly params: tf.Variable[] = [];
  private readonly optimizer: tf.Optimizer;
  private steps = 0;
  /** Rank of each frequency from slowest (0) to fastest, for coarse-to-fine. */
  private readonly frequencyRank: number[] = [];
  /** Per-feature weights in [0, 1] while frequencies are being unlocked; null when all are on. */
  private frequencyMask: tf.Tensor2D | null = null;

  /**
   * @param images the picture to learn, shape [H, W, 3] with values in [0, 1];
   *   or several same-size pictures, to learn color = f(x, y, t) with picture i at t = i / (n - 1).
   */
  constructor(images: tf.Tensor3D | tf.Tensor3D[], config: ModelConfig) {
    this.config = config;
    const pictures = Array.isArray(images) ? images : [images];
    this.frames = pictures.length;
    const [height, width] = pictures[0].shape;
    this.target = tf.tidy(() => tf.concat(pictures.map((p) => p.reshape([height * width, 3]))) as tf.Tensor2D);

    // The input never changes, so encode it once instead of on every step.
    this.B = config.fourier
      ? tf.randomNormal<tf.Rank.R2>([2, config.frequencies], 0, config.sigma, 'float32', config.seed)
      : null;
    this.encoded = tf.tidy(() => this.encode(pixelCoordinates(width, height)));
    this.inputs =
      this.frames === 1
        ? this.encoded
        : tf.tidy(() => tf.concat(pictures.map((_, i) => this.withTime(this.encoded, i / (this.frames - 1)))));

    // A stack of dense layers: ReLU in the middle, 3 outputs (r, g, b) at the end.
    const sizes = [this.inputs.shape[1], ...Array(config.hiddenLayers).fill(config.hiddenUnits), 3];
    for (let i = 0; i < sizes.length - 1; i++) {
      const limit = Math.sqrt(6 / sizes[i]); // He-uniform init, good for ReLU
      const w = tf.randomUniform([sizes[i], sizes[i + 1]], -limit, limit, 'float32', config.seed + i + 1);
      this.params.push(tf.variable(w), tf.variable(tf.zeros([sizes[i + 1]])));
      w.dispose();
    }
    this.parameterCount = this.params.reduce((n, p) => n + p.size, 0);
    this.optimizer = tf.train.adam(config.learningRate);

    if (this.B) {
      const b = this.B.dataSync();
      const m = config.frequencies;
      const order = Array.from({ length: m }, (_, k) => k).sort(
        (i, j) => Math.hypot(b[i], b[m + i]) - Math.hypot(b[j], b[m + j]),
      );
      order.forEach((k, rank) => (this.frequencyRank[k] = rank));
      this.updateSchedule();
    }
  }

  /**
   * Per-input weights from a weight per frequency rank (0 = slowest): sin and
   * cos of a frequency share one weight, and t (if any) always passes through.
   */
  private frequencyWeights(weight: (rank: number) => number): tf.Tensor2D {
    const m = this.config.frequencies;
    const weights = new Float32Array(this.inputs.shape[1]).fill(1);
    this.frequencyRank.forEach((rank, k) => (weights[k] = weights[m + k] = weight(rank)));
    return tf.tensor2d(weights, [1, weights.length]);
  }

  /**
   * Training schedules, applied before every step:
   * - learning rate: big steps early to get the shapes, small steps late to settle the detail;
   * - coarse-to-fine: frequencies fade in from slowest to fastest, so the network
   *   lays down the broad shapes before it is handed the fine stripes.
   */
  private updateSchedule(): void {
    const { decayStart, decaySteps, learningRate, finalLearningRate, coarseToFineSteps, frequencies: m } = this.config;
    if (decaySteps > 0) {
      const progress = Math.min(Math.max(this.steps - decayStart, 0) / decaySteps, 1);
      // Adam reads this field on every update.
      (this.optimizer as unknown as { learningRate: number }).learningRate =
        learningRate * (finalLearningRate / learningRate) ** progress;
    }
    const unlocking = this.frequencyMask || this.steps === 0;
    if (this.frequencyRank.length && coarseToFineSteps > 0 && unlocking && this.steps % 10 === 0) {
      this.frequencyMask?.dispose();
      this.frequencyMask = null;
      if (this.steps < coarseToFineSteps) {
        // Start with the slowest eighth of the frequencies, end with all of them;
        // each frequency fades in over one rank.
        const unlocked = m / 8 + ((7 * m) / 8) * (this.steps / coarseToFineSteps);
        this.frequencyMask = this.frequencyWeights((rank) => Math.min(Math.max(unlocked - rank, 0), 1));
      }
    }
  }

  /** What the network actually receives for a coordinate: γ(x, y), or plain (x, y). */
  private encode(coords: tf.Tensor2D): tf.Tensor2D {
    return this.B ? fourierFeatures(coords, this.B) : coords.clone();
  }

  /** Append t as one more input column (only when learning several pictures). */
  private withTime(encoded: tf.Tensor2D, t: number): tf.Tensor2D {
    if (this.frames === 1) return encoded;
    return tf.concat([encoded, tf.fill([encoded.shape[0], 1], t)], 1) as tf.Tensor2D;
  }

  /** The function itself: encoded coordinates in, colors out. */
  private forward(inputs: tf.Tensor2D, mask = this.frequencyMask): tf.Tensor2D {
    return tf.tidy(() => {
      let h: tf.Tensor2D = mask ? inputs.mul(mask) : inputs;
      for (let i = 0; i < this.params.length; i += 2) {
        h = h.matMul(this.params[i] as tf.Tensor2D).add(this.params[i + 1]);
        h = i + 2 < this.params.length ? h.relu() : h.sigmoid();
      }
      return h;
    });
  }

  /** Everything training compares against the originals: every pixel of every picture. */
  predict(): tf.Tensor2D {
    return this.forward(this.inputs);
  }

  /** The network's painting at t (any value, not just the trained ones): shape [H*W, 3], values in [0, 1]. */
  paint(t = 0, visibleFrequencies = this.config.frequencies): tf.Tensor2D {
    return tf.tidy(() => {
      // Optionally hide all but the k slowest waves (display only: training always sees them all).
      const mask =
        this.frequencyRank.length && visibleFrequencies < this.config.frequencies
          ? this.frequencyWeights((rank) => (rank < visibleFrequencies ? 1 : 0))
          : this.frequencyMask;
      return this.forward(this.frames === 1 ? this.inputs : this.withTime(this.encoded, t), mask);
    });
  }

  /** Paint and score in one pass: the colors plus the share of pixels painted right. */
  evaluate(tolerance = PIXEL_TOLERANCE): { colors: tf.Tensor2D; accuracy: tf.Scalar } {
    return tf.tidy(() => {
      const colors = this.predict();
      const worstChannel = colors.sub(this.target).abs().max(1);
      const accuracy = worstChannel.lessEqual(tolerance).toFloat().mean() as tf.Scalar;
      return { colors, accuracy };
    });
  }

  /** Mean of the encoded input, summed in double precision on the CPU (for diagnostics). */
  async inputMean(): Promise<number> {
    const values = await this.inputs.data();
    let sum = 0;
    for (const v of values) sum += v;
    return sum / values.length;
  }

  /** Call the function at any point, even between pixels. Returns [r, g, b] in [0, 1]. */
  colorAt(x: number, y: number, t = 0): tf.Tensor1D {
    return tf.tidy(() => this.forward(this.withTime(this.encode(tf.tensor2d([[x, y]])), t)).squeeze() as tf.Tensor1D);
  }

  /**
   * One training step: paint everything, measure the error (mean squared
   * difference from the original), and let Adam nudge every weight downhill.
   * Returns the error only when asked, because reading it back from the GPU
   * forces a sync and slows the loop.
   */
  step(returnLoss = false): tf.Scalar | null {
    this.updateSchedule();
    this.steps++;
    const loss = () => tf.losses.meanSquaredError(this.target, this.predict()) as tf.Scalar;
    return this.optimizer.minimize(loss, returnLoss, this.params);
  }

  dispose(): void {
    tf.dispose([this.encoded, this.inputs, this.target, ...this.params]);
    this.B?.dispose();
    this.frequencyMask?.dispose();
    this.optimizer.dispose();
  }
}

/** Peak signal-to-noise ratio in dB, the usual "how close is the image" score. */
export const psnr = (mse: number): number => -10 * Math.log10(mse);
