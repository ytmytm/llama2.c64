# Llama2.c64

This is a [llama2.c](https://github.com/karpathy/llama2.c) port to the C64 equipped with the [260K tinystories model](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K).

You can recompile it to run your own model. As long as it fits within 2MB (real hardware) or 16MB (virtual like UII+) together with caches, it should work fine.

This project is a port of the Llama2.c codebase to the Commodore 64, hence the name `llama2.c64`. The goal is to bring the functionality of Llama2 to the C64 platform, leveraging its unique hardware capabilities.

# Screenshots

## Parameter screen

**TODO**

## Prompt and output

**TODO**

# How to run it with VICE?

Enable REU, set REU size to 2MB, and set REU image to the provided `weights.bin`. Load the program and turn on warp.

```
x64 -warp -reu -reusize 2048 -reuimage weights.bin llama2c64.prg
```

## Settings screenshot

**TODO**

# Pros

- Low power consumption
- On-premise
- Safe: **your** data is completely under **your** control, it's not used to train new models
- Doesn't require an expensive GPU
- Waiting for the next token on a C64 is just as exciting as waiting for one coming from DeepSeek running on your laptop

# Cons

- None really, this is fantastic
- Ram Expansion Unit (REU) with at least 2MB is necessary
- Feels a bit slow, not for the impatient

# Credits

Ported to C64 by Maciej 'YTM/Elysium' Witkowiak using [oscar64](https://github.com/drmortalwombat/oscar64)

# Technical details

## Model

There are two parts to the model: tokenizer and model weights. For C64, they need to be processed a bit. This can be done with the `generate-model-files.py` script.

The script will read the tokenizer and model weights and save the corresponding files:

- tokenizer.bin - tokenizer data with NULL-terminated strings, uint16_t vocabulary size and offsets, and with uint8_t string lengths
- config.bin - model parameters
- weights.bin - model weights, a REU image padded to the next valid size (2MB, 4MB, 16MB)

Original model weights and tokenizer file came from the [tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) repository. You will find there also training information.

## Verification against `llama2.c`

Run llama2.c in deterministic mode (temperature=0.0) and try the same prompt on C64:
```
./run stories260K.bin -z tok512.bin -t 0 -i "Zoo" -n 60
```

In both cases, you should see
```
Zoo was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but she didn't want to play with
```

## Memory

- The tokenizer and its encoding/decoding dictionaries fit within C64 memory (`tokenizer64.c`)
- Model weights and some of the data structures have to be in REU due to their size (`transformer.c`)
- Some remaining data structures from `Transformer` also stayed within C64 memory

## `math.c`

I provide my own code for `my_sin`, `my_cos`, and `my_exp` for better accuracy than the ones that come with [oscar64](https://github.com/drmortalwombat/oscar64).
My polynomial factors are actually copied from C64 BASIC.

## Branches

- `wrapped_debug` - development branch with lots of debug messages and data structure dumps for calculation comparisons with `llama2.c`, use that as a start for the quantized version; it also shows how much memory is used for each part (note: top-p was not backported there)
- `feature-fastmult` - an attempt to speed up `float32` multiplication using `uint8_t` times table (64K in REU); it turned out to be twice as slow, but nevertheless can be useful for the quantized version

# FAQ

## Is this a joke?

No, it really runs the same set of calculations as [llama2.c](https://github.com/karpathy/llama2.c) and returns exactly the same results. Just on a humble C64.

There is plenty of information provided in the README of [llama2.c](https://github.com/karpathy/llama2.c).
You can [read more about Transformer models here](https://medium.com/@smmzhu/demystifying-the-transformer-model-cd73e1b7ac87).

## What's the performance like? I have been waiting here for 15 minutes and it does nothing

You will receive one output token about every 8 minutes. Note that the very first token is a start marker, so the text in the output will start appearing after 16 minutes.

## What do those parameters mean?

- `temperature` controls the randomness of the output, if set to `0.0` the result is deterministic
- `top-p` ensures that tokens with tiny probabilities do not get sampled. Lower values make the output more focused and deterministic, while higher values increase diversity, if set to `0.0` the feature is off
- `output tokens` controls the number of output tokens, note that one token may be more than one letter (e.g. `was` or `once` are tokens in the `tinystories` model)

*To control the diversity of samples, use either the temperature or the top-p value, but not both. Vary the temperature between 0.0 and 1.0 and keep top-p off (set to 0.0), or vary the top-p value between 0.0 and 1.0 and keep the temperature at 1.0.*

## What is the yellow number under the clock?

That's the number of input tokens. In this model, the input tokens are first all copied to output before any sampling happens.

## How to compile it?

Usually, I provide a `Makefile` but `oscar64` wants everything in one file, so it's just:
```
oscar64 -O2 llama2c64.c
```
Do not use other optimizations besides `-O0`, `-O1`, or `-O2`, they break the program.

## Can it run faster?

Yes, but just a bit. Several things can be optimized, but the truth is - it doesn't matter. The program spends most of the time in one of the three matrix multiplication functions. Optimizing anything else is a waste of time.

## Will it run faster with SCPU?

Certainly, but the results are wrong. I don't know why. (Tested with VICE)

## What about a quantized model?

If you can provide a pull request for a quantized int8 model, be my guest :)
