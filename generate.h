/* Inference for Llama-2 Transformer model in pure C */

#ifndef GENERATE_H
#define GENERATE_H

#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

// generation loop
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps);

#endif // GENERATE_H