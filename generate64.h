/* Inference for Llama-2 Transformer model in pure C */

#ifndef GENERATE_H
#define GENERATE_H

#include <stdint.h>

#include "transformer64.h"
#include "tokenizer64.h"
#include "sampler64.h"

// generation loop
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, uint16_t steps);

#endif // GENERATE_H