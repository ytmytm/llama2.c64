/* Inference for Llama-2 Transformer model in pure C */

#ifndef NNET_H
#define NNET_H

#include "transformer.h"

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// sampler
void softmax(float* x, int size);

// generate
float* forward(Transformer* transformer, int token, int pos);

#endif // NNET_H