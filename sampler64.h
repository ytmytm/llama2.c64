/* Inference for Llama-2 Transformer model in pure C */

#include <stdint.h>

#ifndef SAMPLER_H
#define SAMPLER_H

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    uint16_t index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    uint16_t vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    uint32_t rng_state;
} Sampler;

void build_sampler(Sampler* sampler, uint16_t vocab_size, float temperature, float topp, uint32_t rng_seed);
void free_sampler(Sampler* sampler);

// generate.c
// sample the token given the logits and some hyperparameters
uint16_t sample(Sampler* sampler, float* logits);

#endif // SAMPLER_H
