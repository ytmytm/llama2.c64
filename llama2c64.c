/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer64.h"
#include "transformer64.h"
#include "nnet64.h"
#include "sampler64.h"
#include "util.h"
#include "generate64.h"
#include "ui64.c"
#include "math.c"
#include "tokenizer64.c"
#include "transformer64.c"
#include "nnet64.c"
#include "sampler64.c"
#include "util64.c"
#include "generate64.c"

#include <c64/memmap.h>
#include <conio.h>

#pragma region( main, 0x0a00, 0xd000, , , {code, data, bss, heap, stack} )

//#pragma stacksize(4096)
//#pragma heapsize(8192)

int main(void) {

    mmap_set(MMAP_NO_BASIC);

    // default parameters
    float temperature = 0.0;    // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9;           // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 64;            // number of steps to run for

    // parameter validation/overrides
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    load_tokenizer(&tokenizer);

    Transformer transformer;
    load_transformer(&transformer);
    Config64 *c = transformer.config;

    Config64* p = transformer.config;
    TransformerWeights64* w = &transformer.weights;
    RunState64* s = &transformer.state;

    Sampler sampler;
    build_sampler(&sampler, c->vocab_size, temperature, topp, 123456);

    ui_init();
    char *prompt = malloc(256);
    ui_get_prompt(prompt);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    while (true) {

    }

    return 0;
}

