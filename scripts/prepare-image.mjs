/**
 * One-off: turn the full-size print into the small training target.
 *
 *   npm run prepare-image -- source/great-wave.jpg
 *
 * Keeps the print's proportions (the wave *and* Fuji) and writes a 128 px wide
 * PNG to public/. The page never fetches the painting at runtime.
 */
import sharp from 'sharp';

const [input = 'source/great-wave.jpg', width = '128'] = process.argv.slice(2);
const output = 'public/great-wave.png';

// Trim a thin border so the paper edge and scan margins don't dominate the error.
const meta = await sharp(input).metadata();
const inset = Math.round(meta.width * 0.01);

const info = await sharp(input)
  .extract({ left: inset, top: inset, width: meta.width - 2 * inset, height: meta.height - 2 * inset })
  .resize({ width: Number(width), kernel: 'lanczos3' })
  .removeAlpha()
  .png()
  .toFile(output);

console.log(`${input} (${meta.width}×${meta.height}) -> ${output} (${info.width}×${info.height}, ${info.width * info.height} pixels)`);
