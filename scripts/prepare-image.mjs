/**
 * One-off: turn a full-size print into a small training target.
 *
 *   npm run prepare-image -- source/great-wave.jpg public/great-wave.png
 *   npm run prepare-image -- source/red-fuji.jpg public/red-fuji.png --crop=42,33,1202,819 --size=128x85
 *
 * --crop=L,T,W,H cut the print out of its paper margin first (measured by hand)
 * --size=WxH force an exact size (cropping to fill), so two images share one pixel grid
 * Default: 128 px wide, keeping the print's proportions.
 * The page never fetches the paintings at runtime.
 */
import sharp from 'sharp';

const args = process.argv.slice(2);
const [input = 'source/great-wave.jpg', output = 'public/great-wave.png'] = args.filter((a) => !a.startsWith('--'));
const crop = args.find((a) => a.startsWith('--crop='))?.slice(7).split(',').map(Number);
const size = args.find((a) => a.startsWith('--size='))?.slice(7).split('x').map(Number);

let image = sharp(input);
if (crop) {
  const [left, top, width, height] = crop;
  image = sharp(await image.extract({ left, top, width, height }).png().toBuffer());
}
const meta = await image.metadata();

// Trim a thin border so the paper edge and scan margins don't dominate the error.
const inset = Math.round(meta.width * 0.01);
const info = await image
  .extract({ left: inset, top: inset, width: meta.width - 2 * inset, height: meta.height - 2 * inset })
  .resize(size ? { width: size[0], height: size[1], fit: 'cover', kernel: 'lanczos3' } : { width: 128, kernel: 'lanczos3' })
  .removeAlpha()
  .png()
  .toFile(output);

console.log(`${input} (${meta.width}×${meta.height}${crop ? ' after crop' : ''}) -> ${output} (${info.width}×${info.height})`);
