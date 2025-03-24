/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LUT_OFFSET (0x1F0000)

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

#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/memmap.h>
#include <conio.h>

#pragma region( main, 0x0a00, 0xd000, , , {code, data, bss, heap, stack} )

//#pragma stacksize(4096)
//#pragma heapsize(8192)

int main(void) {

    mmap_set(MMAP_NO_BASIC);

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

    ui_init();

    ui_startup_screen(c);

    ui_inference_screen_init();
    char *prompt = malloc(256);
    ui_get_prompt(prompt);

    uint32_t seed;
    seed = cia1.ta << 16 | vic.raster | cia2.todt; 
    build_sampler(&sampler, c->vocab_size, temperature, topp, seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    while (true) {

    }

    return 0;
}

