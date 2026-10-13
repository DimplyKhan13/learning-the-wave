/**
 * Produce the reference numbers the browser's ?diagnose=1 page compares against.
 *
 *   npx tsx scripts/diagnose.mts
 */
import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import { writeFileSync } from 'node:fs';
import { DEFAULT_CONFIG } from '../src/model.ts';
import { diagnose } from '../src/diagnose.ts';

const { data, info } = await sharp('public/great-wave.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
const pixels = tf.tidy(() => tf.tensor3d(new Uint8Array(data), [info.height, info.width, 3]).toFloat().div(255) as tf.Tensor3D);
const report = await diagnose(pixels, DEFAULT_CONFIG);
writeFileSync('src/diagnose-reference.json', JSON.stringify(report, null, 2) + '\n');
console.log(report);
