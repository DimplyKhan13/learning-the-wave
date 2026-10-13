# Learning the Wave

A neural network that never sees *The Great Wave off Kanagawa* learns to paint it, live in your browser.

It gets a pixel's coordinate `(x, y)`, guesses a color, and is told how wrong it was. Repeat a few thousand times and
the print appears. No server, no Python: TypeScript and TensorFlow.js in a browser tab.

**Live demo and write-up:** https://dimplykhan13.github.io/learning-the-wave/

## What's in the page

- **The demo:** the original, what the network paints, and the share of pixels it gets right, updated live.
  Switch Fourier features off to watch it get stuck in a blur.
- **The equation:** the whole network written as one function, and the striped inputs it actually receives.
  Hover the painting to evaluate `f(x, y)` at any point.
- **Waves the network can see:** hide the fastest waves and see that the network mixes them rather than adding them up.
- **One function, two prints:** `color = f(x, y, t)` learns the Great Wave at `t = 0` and Red Fuji at `t = 1`.
  Type a four-letter word to pick `t`.
- **Diagnostics:** a seeded training run compared number by number against a CPU reference. This is how a WebGL
  precision problem on a real GPU was found; the page now prefers WebGPU.

## The model

- Input: `(x, y)` in `[-1, 1]`, encoded as Fourier features `γ(v) = [sin(2πBv), cos(2πBv)]`
  ([Tancik et al., 2020](https://arxiv.org/abs/2006.10739)), with 128 fixed random frequencies (σ = 8).
- MLP: 3 hidden layers of 128 ReLU units, sigmoid output for `(r, g, b)`. 66,307 parameters.
- Loss: mean squared error over all 10,880 pixels (128 × 85) every step. Adam, learning rate 1e-3, fixed seed.

## Run it

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/
```

Useful extras:

```sh
npm run benchmark -- 5000 --tolerance=0.01   # train with tfjs-node, print accuracy milestones
npm run diagnose                             # regenerate the reference numbers for the diagnostics
npm run prepare-image -- source/great-wave.jpg public/great-wave.png
```

`?backend=webgpu|webgl|wasm|cpu` forces a TensorFlow.js backend; `?chart=error` shows the raw error chart instead of
the accuracy chart; `?diagnose=1` runs the diagnostics on load.

## Layout

| File | What it does |
| --- | --- |
| `src/model.ts` | The network: Fourier features, the MLP, one training step. The core of the project. |
| `src/trainer.ts` | The training loop: frame budget, repaint every N steps, pause when the tab is hidden. |
| `src/chart.ts` | The charts, drawn by hand on a `<canvas>`. |
| `src/equation.ts` | The formula and the striped input tiles. |
| `src/morph.ts`, `src/words.ts` | `f(x, y, t)` and the four-letter word rule. |
| `src/diagnose.ts` | The seeded cross-backend check. |
| `src/main.ts` | Page wiring. |
| `scripts/` | Image preparation, benchmark, diagnostics reference. |

## Credits

*The Great Wave off Kanagawa* and *Fine Wind, Clear Morning* (Red Fuji), Katsushika Hokusai, c. 1831.
The Metropolitan Museum of Art, public domain.
