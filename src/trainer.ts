/**
 * Runs the training loop without freezing the page.
 *
 * Each animation frame runs a small batch of steps, reads the error back once,
 * and repaints the network's picture every `renderEvery` steps. The batch size
 * adapts so that a frame's work stays within a time budget on slow and fast devices alike.
 */
import * as tf from '@tensorflow/tfjs';
import { PaintingNet, type ModelConfig } from './model';

export interface TrainerEvents {
  /** Called with the latest error after every frame of training. */
  onLoss(step: number, loss: number): void;
  /**
   * Called with the network's current painting, shape [H, W, 3], and the share
   * of pixels it paints right (0–1).
   */
  onPaint(step: number, image: tf.Tensor3D, accuracy: number): Promise<void>;
  /** Called whenever play/pause state or speed changes. */
  onStatus(status: { running: boolean; step: number; stepsPerSecond: number }): void;
}

const FRAME_BUDGET_MS = 30;

export class Trainer {
  step = 0;
  running = false;
  stepsPerSecond = 0;
  /** Repaint the network's output every N steps (repainting costs a GPU readback). */
  renderEvery = 10;

  /** The network being trained, for anyone who wants to call it (e.g. the hover readout). */
  net: PaintingNet | null = null;
  private stepsPerFrame = 1;
  private lastPainted = -1;
  private generation = 0; // bumps on restart, so a stale loop knows to stop
  private resumeWhenVisible = false;

  constructor(
    private readonly pixels: tf.Tensor3D,
    private readonly events: TrainerEvents,
  ) {
    // No one is watching a hidden tab: stop burning their battery and GPU.
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.running) {
        this.resumeWhenVisible = true;
        this.pause();
      } else if (!document.hidden && this.resumeWhenVisible) {
        this.resumeWhenVisible = false;
        this.play();
      }
    });
  }

  /** Throw away the current network and start from random weights. */
  async restart(config: ModelConfig): Promise<void> {
    this.generation++;
    this.net?.dispose();
    this.net = new PaintingNet(this.pixels, config);
    this.step = 0;
    this.lastPainted = -1;
    this.stepsPerFrame = 1;
    await this.paint();
    this.emitStatus();
  }

  play(): void {
    if (this.running || !this.net) return;
    this.running = true;
    this.emitStatus();
    void this.loop(this.generation);
  }

  pause(): void {
    this.running = false;
    this.emitStatus();
  }

  /** Train up to exactly `target` steps, then paint. Used by the video recorder. */
  async runTo(target: number): Promise<number> {
    if (!this.net) throw new Error('call restart() first');
    const chunk = 25;
    while (this.step < target) {
      const n = Math.min(chunk, target - this.step);
      for (let i = 0; i < n - 1; i++) this.net.step();
      const loss = this.net.step(true)!;
      this.step += n;
      const value = (await loss.data())[0];
      loss.dispose();
      this.events.onLoss(this.step, value);
    }
    await this.paint();
    this.emitStatus();
    return this.step;
  }

  private async loop(generation: number): Promise<void> {
    while (this.running && generation === this.generation && this.net) {
      const started = performance.now();

      // Queue up a batch of steps on the GPU; only the last one reports its error.
      for (let i = 0; i < this.stepsPerFrame - 1; i++) this.net.step();
      const loss = this.net.step(true)!;
      this.step += this.stepsPerFrame;
      const value = (await loss.data())[0]; // waits for the GPU to finish the batch
      loss.dispose();
      if (generation !== this.generation) return;
      this.events.onLoss(this.step, value);

      if (this.step - this.lastPainted >= this.renderEvery) await this.paint();

      // Adapt the batch so a frame stays near the time budget.
      const elapsed = performance.now() - started;
      const perStep = elapsed / this.stepsPerFrame;
      this.stepsPerSecond = 0.8 * this.stepsPerSecond + 0.2 * (1000 / perStep);
      this.stepsPerFrame = Math.max(1, Math.min(50, Math.round(FRAME_BUDGET_MS / perStep)));
      this.emitStatus();

      await new Promise(requestAnimationFrame); // give the browser a chance to draw
    }
  }

  private async paint(): Promise<void> {
    if (!this.net) return;
    const [height, width] = this.pixels.shape;
    const { colors, accuracy } = this.net.evaluate();
    const image = colors.reshape([height, width, 3]) as tf.Tensor3D;
    try {
      await this.events.onPaint(this.step, image, (await accuracy.data())[0]);
    } finally {
      tf.dispose([colors, accuracy, image]);
    }
    this.lastPainted = this.step;
  }

  private emitStatus(): void {
    this.events.onStatus({ running: this.running, step: this.step, stepsPerSecond: this.stepsPerSecond });
  }
}
