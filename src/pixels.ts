import * as tf from '@tensorflow/tfjs';

/**
 * An image's pixels as numbers in [0, 1], shape [H, W, 3], at the file's own
 * size. (tf.browser.fromPixels on an <img> uses its on-screen size instead,
 * which changes with the layout.)
 */
export function readPixels(img: HTMLImageElement): tf.Tensor3D {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d')!.drawImage(img, 0, 0);
  return tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().div(255) as tf.Tensor3D);
}
