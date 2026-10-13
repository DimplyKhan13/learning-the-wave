import { defineConfig } from 'vite';

// Cross-origin isolation lets the WASM backend use threads. GitHub Pages can't
// send these headers, so it only applies locally (dev server, video recording).
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base: './', // works under https://<user>.github.io/learning-the-wave/
  server: { headers: isolation },
  preview: { headers: isolation },
});
