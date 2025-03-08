/* Inference for Llama-2 Transformer model in pure C */

#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdlib.h>

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);

// generate.c
// sample the token given the logits and some hyperparameters
int sample(Sampler* sampler, float* logits);

#endif // SAMPLER_H
