/* Inference for Llama-2 Transformer model in pure C */

// C64 port by Maciej 'YTM/Elysium' Witkowiak, 2025

#ifndef NNET_H
#define NNET_H

#include "transformer64.h"

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// init
void nnet_init(Transformer* transformer);

// sampler
void softmax(REUPtr x, uint16_t size);

// generate
float* forward(Transformer* transformer, uint16_t token, uint16_t pos);

#endif // NNET_H