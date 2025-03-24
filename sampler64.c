/* Inference for Llama-2 Transformer model in pure C */

// C64 port by Maciej 'YTM/Elysium' Witkowiak, 2025

#include <math.h>

#include "sampler64.h"

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

uint16_t sample_argmax(float* probabilities, uint16_t n) {
    // return the index that has the highest probability
    uint16_t max_i = 0;
    float max_p = probabilities[0];
    for (uint16_t i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

uint16_t sample_mult(float* probabilities, uint16_t n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0;
    for (uint16_t i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

/* XXX NOT TESTED YET
void quicksort(ProbIndex* arr, uint16_t left, uint16_t right) {
    uint16_t i = left, j = right;
    ProbIndex tmp;
    ProbIndex pivot = arr[(left + right) / 2];

    // partition
    while (i <= j) {
        while (arr[i].prob > pivot.prob)
            i++;
        while (arr[j].prob < pivot.prob)
            j--;
        if (i <= j) {
            tmp = arr[i]; // XXX does this work for structs?
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    };

    // recursion
    if (left < j)
        quicksort(arr, left, j);
    if (i < right)
        quicksort(arr, i, right);
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    uint16_t n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0 - topp) / (n - 1);
    for (uint16_t i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    quicksort(probindex, 0, n0 - 1);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0;
    uint16_t last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (uint16_t i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0;
    for (uint16_t i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}
*/

void build_sampler(Sampler* sampler, uint16_t vocab_size, float temperature, float topp, uint32_t rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    // can be removed if not using top-p sampling: -p 1.0
// XXX64 bring back if top-p sampling is implemented
//    printf("Allocating sampler->probindex: %zu bytes\n", sampler->vocab_size * sizeof(ProbIndex));
//    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
// XXX64 bring back if top-p sampling is implemented
// free(sampler->probindex);
}

uint32_t random_u32(uint32_t *state) {
    // xorshift* rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

float random_f32(uint32_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0;
}

// same softmax as in nnet64.c, but with local buffer
void softmax_local(float* x, uint16_t size) {
        // find max value (for numerical stability)
        float max_val = x[0];
        for (uint16_t i = 1; i < size; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        // exp and sum
        float sum = 0.0;
        for (uint16_t i = 0; i < size; i++) {
            x[i] = my_exp(x[i] - max_val);
            sum += x[i];
        }
        // normalize
        for (uint16_t i = 0; i < size; i++) {
            x[i] /= sum;
        }
}

uint16_t sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    uint16_t next;
    if (sampler->temperature == 0.0) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (uint16_t q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax_local(logits, sampler->vocab_size); // XXX64: logits are local but softmax works on remote, need local softmax version
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_mult(logits, sampler->vocab_size, coin);
        // XXX64 bring back if top-p sampling is implemented
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
//            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}
