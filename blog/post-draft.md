# Let's teach a neural network to paint, live in your browser

*I wanted to know how AI draws pictures. No pretrained model, no server: you'll watch it learn The Great Wave from scratch.*

<!--
Draft for duarte.dev.br. Style reference: skerritt.blog (short paragraphs,
a chain of "ok, but what if…" sections, emoji callouts, real numbers).

Status per section:
  [approved] Intro
  [draft]    A picture is just a function
  [draft]    Ok, but how does it learn?
  [draft]    Why is it a blurry mess? (+ hummingbird callback)
  [todo]     Is it getting better? The chart as a dashboard (+ the drops)
  [todo]     Works on my machine™ (WebGL vs WebGPU diagnostics)
  [todo]     What if I give it a word? f(x, y, t)
  [todo]     Ok, but that's not how the real ones work
  [todo]     Can I zip my JPEGs with this?
  [todo]     What's next?

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

Here's the idea that made it click for me.

A picture is a grid of pixels. But we can also treat it as a question we ask over and over: *what color goes here?*

Give it `(x, y)`, get back `(r, g, b)`.

So instead of storing 10,880 colors, we ask a neural network to *be* that function. A coordinate goes in, a color comes out. That's the whole model.

**[live demo]**

> 👽 **Fun fact:** the network isn't adjusting pixels. It's adjusting one function that paints all of them. That's why the whole picture sharpens at once instead of filling in line by line.

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

---

*Thanks to Himanshu Patel for the hummingbird that started all of this.*
