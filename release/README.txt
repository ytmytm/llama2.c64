# Llama2.c64

## https://github.com/ytmytm/llama2.c64

Ported to C64 by Maciej 'YTM/Elysium' Witkowiak using [oscar64](https://github.com/drmortalwombat/oscar64)

Llama2.c64 is a port of [llama2.c](https://github.com/karpathy/llama2.c) to the Commodore C64 equipped with a 2MB REU.

It runs the [260K tinystories model](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) bringing Llama2's capabilities to the unique C64 hardware environment.

This is not a chat model.

Rather, imagine prompting a 3-year-old child with the beginning of a story — they will continue it to the best of their vocabulary and abilities.

# How to run it?

## VICE

Enable REU, set REU size to 2MB, and set REU image to the provided `weights.reu`. Load the program and turn on warp.

```
x64sc -warp -reu -reusize 2048 -reuimage weights.reu llama2c64.prg
```

## Ultimate II+

Enable REU in the Cartridge settings menu, with size at least 2MB. Navigate to location with `llama2.c64` and hit `<RETURN>` on `weights.reu` file.
There will be an option to load this image into REU.

Then start `llama2c64.prg`.

# Pros

- Low power consumption
- On-premise inference
- Safe: **your** data is completely under **your** control, it's not used to train new models
- Doesn't require an expensive GPU
- Waiting for the next token on a C64 is just as exciting as waiting for one coming from DeepSeek running on your laptop

# Cons

- None really, this is fantastic
- Ram Expansion Unit (REU) with at least 2MB is necessary
- Feels a bit slow, not for the impatient
- Won't handle models larger than about 8MB, because REU is limited to 16MB

# FAQ

## Is this a joke?

No, it really runs the Llama2 architecture, performing the same set of calculations as [llama2.c](https://github.com/karpathy/llama2.c) and producing identical results. A humble C64 runs the Llama2 model - the only limitation being REU memory size.

There is plenty of information provided about this in the README of [llama2.c](https://github.com/karpathy/llama2.c).
You can [read more about Transformer models here](https://medium.com/@smmzhu/demystifying-the-transformer-model-cd73e1b7ac87).

## Verification against `llama2.c`

Run llama2.c in deterministic mode (temperature=0.0) and try the same prompt on C64:

For example:
```
./run stories260K.bin -z tok512.bin -t 0 -i "Zoo" -n 60
```

In both cases, you should see
```
Zoo was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but she didn't want to play with
```

## What's the performance like? I have been waiting here for 15 minutes and it does nothing

You will receive one output token approximately every 8 minutes. This is an estimation, the attention step depends on the number of tokens generated so far, so the process gets slower and slower.

The very first produced token is a start marker, so the text in the output will start appearing after 16 minutes. All the tokens from the input will be repeated in the output before any sampling starts.

## What do those parameters mean?

- `temperature` controls the randomness of the output, if set to `0.0` the result is deterministic
- `top-p` ensures that tokens with tiny probabilities do not get sampled. Lower values make the output more focused and deterministic, while higher values increase diversity, if set to `0.0` the feature is off. This setting has no effect if temperature is `0.0`
- `output tokens` controls the number of output tokens, one token may be more than one letter (e.g. `was` or `once` are tokens in the `tinystories` model); note that it's just a stop condition, it doesn't control the verbosity of the model

*To control the diversity of samples, use either the temperature or the top-p value, but not both. Vary the temperature between 0.0 and 1.0 and keep top-p off (set to 0.0), or vary the top-p value between 0.0 and 1.0 and keep the temperature at 1.0.*

## What is the yellow number under the clock?

That's the number of input tokens. In this model, the input tokens are first all copied to output before any sampling happens.

## Can it run faster?

Yes, but just a bit. Several things can be optimized, but the truth is - it doesn't matter. The program spends most of the time in one of the three matrix multiplication functions. Optimizing anything else is a waste of time.

## Will it run faster with SCPU?

Certainly faster, but the results are wrong when SCPU is in turbo mode. I didn't investigate why. (Tested with VICE)

## Can I chat with it?

No, it's not a model instructed and trained for chat. It can only tell a short story.

## Clock doesn't advance

Your CIA is broken or 9VAC is missing.
