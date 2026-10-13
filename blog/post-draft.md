# Let's teach a neural network to paint, live in your browser

*I wanted to know how AI draws pictures. No pretrained model, no server: you'll watch it learn The Great Wave from scratch.*

> 💡 Built with an AI pair programmer. Every number in this post was measured.

<!--
Draft for duarte.dev.br. Style reference: skerritt.blog (short paragraphs,
a chain of "ok, but what if…" sections, emoji callouts, real numbers).

Status per section:
  [approved] Title, subtitle, AI note, intro (up to "Let's build it.")
  [draft]    A picture is just a function
  [draft]    Ok, but how does it learn?
  [draft]    Why is it a blurry mess? (+ hummingbird callback)
  [draft]    Is it actually getting better? (the chart, the drops, failed speedups)
  [draft]    Works on my machine™ (WebGL vs WebGPU diagnostics)
  [draft]    What if I give it a word? f(x, y, t)
  [draft]    Ok, but that's not how the real ones work
  [draft]    Can I zip my JPEGs with this?
  [draft]    What's next?

[ARTHUR: …] marks something only Arthur can supply.

Open questions: tone of the drafted sections; AI transparency line at the top
(yes/no); "Add waves one by one" animation for the slider.
Every number below was measured in this project; don't add any that weren't.
-->

It started with a hummingbird.

Someone on my LinkedIn feed, [Himanshu Patel](https://lnkd.in/p/dZBkxuXH), posted an animation: two chains of circles spinning on top of circles, one steering a pen left and right, the other up and down. With one wave, the pen drew a rough blob. Then another wave joined, and another. By wave 100, the outline of a hummingbird was sharp down to the feather tips. The caption said: *"Sharp feather tips from smooth waves."*

I watched it more times than I'd like to admit.

Then the obvious next question showed up: if a few spinning circles can draw a bird, how do AI tools draw *anything*?

I didn't want the 40-page answer. I wanted the smallest piece I could build myself, run, and actually understand. So I picked a picture, *The Great Wave off Kanagawa* (if you're going to be wrong about something, be wrong in front of a masterpiece), and set one rule:

**The network never gets to see it.**

Let's build it.

## A picture is just a function

First, the spoiler: AI is math.

Remember `y = ax + b`? A neural network is that, stacked: multiply, add, repeat. Ours has 66,307 `a`s and `b`s instead of two, and training is the hunt for good values.

A formula. Numbers in, numbers out.

**[the equation panel]**

> 🤓 **Nerd note: reading the formula**
> - **`W`** (weights): grids of the `a`s from `y = ax + b`. Multiplying by `W` mixes every input into every output.
> - **`b`** (biases): the `b`s. A nudge added after each multiply.
> - **`relu`**: "keep only the positive part". Negative numbers become 0. This tiny bend is what lets stacked layers draw curves instead of one straight line.
> - **`sigmoid`**: squeezes the final numbers into 0–1, so every output is a valid color.
> - **`γ(x, y)`**: how we feed in the coordinate. Spoiler: it's circles. We'll get there.
> - **`B`**: 128 random frequencies, picked once and never trained.
>
> Training only ever changes the `W`s and `b`s. That's the 66,307 numbers.

And a picture? Ask it a question: *what color goes at `(x, y)`?* Numbers in, numbers out. So let's make our formula *be* the picture.

**[live demo]**

> 👽 **Fun fact:** it isn't adjusting pixels. It's adjusting one formula that paints all of them. That's why the whole picture sharpens at once.

## Ok, but how does it learn?

At the start our network knows nothing, so it paints noise. Confident, colorful noise.

Then we make it play a very simple game, a few thousand times:

1. **Guess.** Paint every pixel.
2. **Measure.** Compare with the original. How wrong was it, on average?
3. **Nudge.** Move each of its 66,307 numbers a tiny bit in the direction that makes it less wrong.
4. **Repeat.**

That's training. No rules about waves or mountains anywhere. Just *guess, measure, nudge, repeat*.

The chart next to the painting is the part I care about most. Without that one number, the loop has nothing to follow. Same as in production: if you can't measure the error, nothing gets better.

## Why is it a blurry mess?

Switch the demo to **Without** and watch.

It gets the big shapes fast: dark water at the bottom, pale sky at the top, a smudge where the wave should be. And then… it just stops. Thousands of steps later it's still a smudge. In my runs, after 3,000 steps its error was still more than 100 times higher than the other version reaches in under 1,000.

That's not a bug. Neural networks love smooth things (there's a name for it: *spectral bias*), and the sharp edges of a woodblock print are about as far from smooth as it gets.

### Remember the hummingbird?

*"Sharp feather tips from smooth waves"* turns out to be exactly our problem. And the fix is the same trick, turned sideways.

The hummingbird is a line: one input, `t`, and every new circle spins faster than the last. Our painting is a surface: two inputs, `x` and `y`. So instead of circles in a neat row, we give the network 128 circles pointing in random directions at random speeds. For every pixel we ask: *where is the point on this circle, at this position?* The answer is a `sin` and a `cos`: 256 numbers per pixel instead of 2.

Seen across the whole image, each circle becomes a striped pattern:

**[striped tiles]**

Now fine detail is just a matter of combining stripes.

### Let's add the waves one at a time, too

That animation stuck with me, so let's do the same thing with our painting. Drag the slider from 1 to 128 waves:

**[slider: waves the network can see]**

And here's the hummingbird's method, plain adding, with the same waves:

**[comparison: network vs. plain sum]**

> 🤔 **This is the part that surprised me most.** His hummingbird needs 100 waves to look sharp. Our painting gets 128, and adding them up still gives you a blur. The network takes the very same waves and *mixes* them into the print.

## Is it actually getting better?

Remember the chart? Let's talk about it, because it took me three tries to get it right.

**Try 1: the raw error.** The number training actually minimizes: the average squared difference between our painting and the original, on a log scale. Precise. Also completely unreadable unless you enjoy log axes.

**Try 2: "% of pixels painted right."** Anyone can read that. But what counts as *right*? I started with "every color channel within 5%". On my laptop it hit 99% at step 965, 33 seconds in… and then the chart went flat. The painting kept getting sharper for thousands of steps, and my dashboard said *nothing to see here*.

**Try 3: within 1%.** About ±2.5 levels out of 255, close to what the image file itself can store. Now the chart keeps climbing for as long as there's something left to learn. On my CPU runs: 50% of pixels right by step 980, 90% by 2,450, 95% by 3,460.

Same model, same run, three different stories, depending only on what I chose to measure.

> 🌝 **Fun fact:** this is exactly the alert-threshold problem from production. Too loose and you're blind. Too tight and you get paged for noise. Pick a threshold that still moves when the thing you care about moves.

### What are those cliffs?

Watch the chart for a while and you'll see it: the line climbs, then suddenly drops off a cliff, then climbs right back.

That's not the chart lying. Every so often the optimizer takes a step that's too big, the whole painting shifts a little, and it spends the next few steps correcting. In one run of 2,500 steps I counted 24 drops of more than 10 points. The biggest was 63 points, and at that moment the error jumped to 6× what it had been just before.

A 1% threshold makes these look dramatic, because a tiny shift in brightness pushes thousands of pixels out of tolerance at once. You'd barely notice it in the painting. Your dashboard notices everything.

Could I smooth it out? Taking smaller steps cut the drops from 24 to 4… and the network ended up at 81% instead of 91% at step 2,500. Stable but slow, or fast with hiccups. I kept the hiccups.

> 🥱 **Things I tried that didn't help:**
> - **Shrinking the step size over time.** Worse every way I scheduled it. Shrinking from the start: 45% of pixels right at step 3,000, against 93% with a constant step. Shrinking late: 90–91% at step 4,000, against 96%.
> - **Unlocking the waves slow-to-fast** (coarse-to-fine). It ends up in the same place (94.3% vs 93.0% at step 3,000), just slower to get going.
>
> The plain setup won.

## Works on my machine™

Up to here I'd been testing on a CPU. Then I opened the page on my laptop, with an actual GPU (an RTX 3050), and expected it to fly.

It crawled. 2 to 11 steps per second. And worse, it *learned less per step*: half the pixels right at step ~600, instead of ~160 on the CPU.

Same code. Same seed. Different results. The worst kind of bug.

My first guess was GPU math precision: those `sin` and `cos` circles get big arguments, and GPUs cut corners on `sin()`. So I moved that part to the CPU. Nothing changed. Wrong guess.

So instead of guessing again, I did the production thing: **compare against a known-good reference and find the first number that diverges.**

I built a diagnostics mode into the page. It runs a short training session with a fixed seed and records the error after steps 1, 2, 5, 10, 20, 50 and 100. The same function runs on the CPU (Node.js) to produce the reference, and the page puts them side by side.

| | error after 100 steps | ms per step |
| --- | --- | --- |
| Reference (CPU) | 0.00598 | 76 |
| WebGL (RTX 3050) | 0.0294 (**~5×**) | 86 |
| WebGPU (RTX 3050) | 0.00597 (**0.2% off**) | 28 |

There it was. Same GPU, same code: WebGL trained to 5× the reference error, WebGPU matched it and ran 3× faster.

So the page now tries WebGPU first, then falls back to WebGL, WebAssembly and plain CPU. On WebGPU my laptop gets half the pixels right (at ±5%) in 7 seconds and 99% in 33.

> ⚠️ I never found out *exactly* what WebGL gets wrong here. I found where it diverges, switched to the path that doesn't, and left a diagnostics button in the page so the next person with a weird GPU can check in 30 seconds. Sometimes that's the job.

## What if I give it a word?

Here's where it starts to look like the AI tools I was curious about.

Our network takes `(x, y)`. Let's give it one more input, `t`, and teach it two pictures at once: *The Great Wave* at `t = 0`, and Hokusai's *Red Fuji* at `t = 1`. Same kind of network, same game. It just has to remember both.

**[demo: f(x, y, t)]**

It works: slide `t` to 0 and you get the wave, slide it to 1 and you get the mountain. It costs something: each step does twice the work, and after 3,000 steps the pair looks a bit worse than the wave did on its own (37.8 dB vs 47.8 dB, if you like that kind of number).

Now, what's at `t = 0.5`?

I was hoping for a morph: the wave rising and freezing into the mountain. What you actually get is closer to a double exposure: both prints overlaid, grainy, the sky turning blue before the wave fades. Which makes sense. We never showed it anything in between, so whatever it paints there is its own guess.

### Words

`t` is just a number, and numbers are boring. So let's type a word instead.

The rule: `WAVE` is 0, `FUJI` is 1, and any four-letter word lands somewhere in between, letter by letter. For each letter, how far is it in the alphabet from the `WAVE` letter and from the `FUJI` letter in the same position? A `Z` in first place is 3 letters from `W` and 20 from `F`, so it leans wave: 3 / (3 + 20) = 0.13. Average the four letters and that's your `t`.

- `WEVE` → 0.05, basically the wave
- `ZZZZ` → 0.43
- `FERI` → 0.63, more mountain than wave

> 🤔 Notice what just happened: **text in, a number out, a picture out.** That's the shape of a text-to-image model. A toy version, sure, minus everything that makes the real ones good. But the same shape.

## Ok, but that's not how the real ones work

Right. Let's be honest about where my little painter stops and the real thing begins.

**What's the same:**
- **A neural network is just a function.** Ours maps `(x, y, t)` to a color. Theirs map much bigger things to much bigger things. Still functions.
- **Training is still guess, measure, nudge, repeat.** At a scale that needs data centers instead of a browser tab, but the same loop.
- **Even the circles show up.** The same `sin`/`cos` trick we used for coordinates is how many big models tell the network *where* something is or *when* it is in a process.

**What's different:**
- **We memorized two pictures. They learn from millions.** Our network can only ever paint the wave or the mountain. A real generator learns what pictures *in general* look like, from huge collections of images paired with captions, so it can paint things nobody showed it.
- **My "text understanding" is the alphabet. Theirs is a language model.** A separate network turns your prompt into a long list of numbers that captures what it *means*, so "a cat on a skateboard" and "a kitten riding a board" land close together. `WAVE` and `WAVY` only land close together in my version by accident.
- **They don't paint coordinates. They clean up noise.** Most of today's generators are *diffusion models*. They start from pure random noise and run a network many times in a row; each pass removes a little noise, nudged by your text, until a picture is left. And that answers our `t = 0.5` problem: they learn what goes *between* pictures because they've seen millions of them.

That's where my understanding currently ends, and also where the next post probably begins.

## Can I zip my JPEGs with this?

The network can repaint the whole print from scratch, so… is it a compressed image?

Not even close:

| | size |
| --- | --- |
| The PNG it learned from | 27.5 KB |
| A JPEG (quality 75) | 3.7 KB |
| Our network's 66,307 numbers | **259 KB** |

Our "compressed" picture is about 70 times bigger than the JPEG. 😅 People do research this seriously, with much smaller networks and tricks to store each number in fewer bits, and it gets competitive at low quality. At 128 × 85 pixels, though, JPEG wins.

What you *do* get is a picture that's a continuous function. Hover over the painting and it tells you the color at any point, including between pixels, at a resolution that never existed in the original.

## What's next?

Like any good side project, the backlog grew faster than the code:

- **Add the waves one by one, as an animation.** A proper answer to the hummingbird: our painting with 1, 2, 3… 128 waves, next to the plain sum, so you can watch the difference instead of dragging a slider.
- **A tiny diffusion model in the browser.** Start from noise, clean it up step by step, and see whether the in-between finally becomes a morph instead of a double exposure.
- [ARTHUR: anything else on your list?]

The code is on GitHub: [github.com/dimplykhan13/learning-the-wave](https://github.com/dimplykhan13/learning-the-wave). If you run it on a strange GPU, hit **Run diagnostics** at the bottom of the page and send me the table. [ARTHUR: how readers should reach you: LinkedIn, email, other?]

---

*Thanks to Himanshu Patel for the hummingbird that started all of this.*

*The Great Wave off Kanagawa* and *Fine Wind, Clear Morning* (Red Fuji), Katsushika Hokusai, c. 1831. The Metropolitan Museum of Art, public domain.
