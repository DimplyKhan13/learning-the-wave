/**
 * Second demo: one function, two prints.
 *
 * The same kind of network learns color = f(x, y, t), with the Great Wave at
 * t = 0 and Red Fuji at t = 1. A slider asks it for any t in between, where it
 * was never trained.
 */
import * as tf from '@tensorflow/tfjs';
import { psnr, type ModelConfig } from './model';
import { Trainer } from './trainer';
import { readPixels } from './pixels';
import { wordToT } from './words';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export interface MorphDemo {
  pause(): void;
}

/**
 * @param images the two prints, already decoded and the same size
 * @param onPlay called when this demo starts training, so the page can pause the other one
 */
export function setupMorph(images: [HTMLImageElement, HTMLImageElement], config: ModelConfig, onPlay: () => void): MorphDemo {
  const canvas = $<HTMLCanvasElement>('morph-painting');
  const badge = $('morph-badge');
  const status = $('morph-status');
  const playButton = $<HTMLButtonElement>('morph-play');
  const slider = $<HTMLInputElement>('morph-t');
  const tLabel = $('morph-t-value');
  canvas.width = images[0].naturalWidth;
  canvas.height = images[0].naturalHeight;

  const pixels = images.map(readPixels);
  let lastLoss = NaN;
  let lastAccuracy = NaN;
  const trainer = new Trainer(pixels, {
    onLoss(_, loss) {
      lastLoss = loss;
    },
    async onPaint(step, image, accuracy) {
      await tf.browser.toPixels(image, canvas);
      lastAccuracy = accuracy;
      badge.textContent = `step ${step.toLocaleString('en-US')}`;
    },
    onStatus({ running, step, stepsPerSecond }) {
      playButton.textContent = running ? 'Pause' : step === 0 ? 'Train' : 'Resume';
      status.textContent = [
        stepsPerSecond && running ? `${stepsPerSecond.toFixed(0)} steps/s` : '',
        `step ${step.toLocaleString('en-US')}`,
        Number.isFinite(lastAccuracy) ? `${(lastAccuracy * 100).toFixed(1)}% of pixels right (both prints)` : '',
        Number.isFinite(lastLoss) ? `PSNR ${psnr(lastLoss).toFixed(1)} dB` : '',
      ]
        .filter(Boolean)
        .join(' · ');
    },
  });

  // Nothing trains until asked: two demos at once would halve the speed of both.
  void trainer.restart(config);
  playButton.disabled = false;
  playButton.addEventListener('click', () => {
    if (trainer.running) return trainer.pause();
    onPlay();
    trainer.play();
  });

  // While paused, repaint on every slider move, but never more than one paint at a time.
  let painting = false;
  let pending = false;
  const repaint = async () => {
    if (painting) return void (pending = true);
    painting = true;
    await trainer.repaint();
    painting = false;
    if (pending) {
      pending = false;
      void repaint();
    }
  };
  const setT = (t: number) => {
    trainer.t = t;
    slider.value = String(t);
    tLabel.textContent = t.toFixed(2);
    if (!trainer.running) void repaint();
  };
  slider.addEventListener('input', () => {
    setT(Number(slider.value));
    wordExplain.textContent = '';
  });

  // Type a four-letter word: a small rule (not the network) turns it into t.
  const wordInput = $<HTMLInputElement>('morph-word');
  const wordExplain = $('morph-word-explain');
  wordInput.addEventListener('input', () => {
    wordInput.value = wordInput.value.toUpperCase().replace(/[^A-Z]/g, '');
    const result = wordToT(wordInput.value);
    if (!result) {
      wordExplain.textContent = 'Four letters, A to Z.';
      return;
    }
    const { t, offWave, offFuji } = result;
    wordExplain.textContent = `${wordInput.value}: ${offWave} letter${offWave === 1 ? '' : 's'} off WAVE, ${offFuji} off FUJI → t = ${offWave} / (${offWave} + ${offFuji}) = ${t.toFixed(2)}`;
    setT(t);
  });

  return { pause: () => trainer.pause() };
}
