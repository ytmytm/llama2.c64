/* Test program for Sampler functions */

#include <stdio.h>
#include "sampler64.h"

void test_sampler(float* logits, uint16_t vocab_size, float temperature, float topp, uint32_t rng_seed, uint16_t expected) {
    Sampler sampler;
    build_sampler(&sampler, vocab_size, temperature, topp, rng_seed);
    uint16_t result = sample(&sampler, logits);
    printf("TEMP: %.1f, SEED: %d, EXPECTED: %d, RESULT: %d\n", temperature, (uint16_t)rng_seed, expected, result);
    free_sampler(&sampler);
}

void test_random_numbers(uint32_t rng_seed, float* expected_values) {
    uint32_t state = rng_seed;
    printf("RNG SEED: %d\n", rng_seed);
    for (int i = 0; i < 5; i++) {
        float random_value = random_f32(&state);
        printf("EXPECTED: %f, RESULT: %f\n", expected_values[i], random_value);
    }
}

int main(void) {
    uint16_t vocab_size = 5;
    float logits[] = {0.1, 0.2, 0.3, 0.25, 0.15}; // Random numbers that sum up to 1.0

    // Test with temperature = 0 (greedy argmax)
    test_sampler(logits, vocab_size, 0.0, 1.0, 12345, 2); // Expected: 2 (highest probability)

    // Test with temperature = 1.0 and different RNG seeds
    test_sampler(logits, vocab_size, 1.0, 1.0, 12345, 3); // Expected: 2 (based on RNG seed)
    test_sampler(logits, vocab_size, 1.0, 1.0, 54321, 1); // Expected: 3 (based on RNG seed)

    // Test with temperature = 0.5 and different RNG seeds
    test_sampler(logits, vocab_size, 0.5, 1.0, 12345, 3); // Expected: 2 (based on RNG seed)
    test_sampler(logits, vocab_size, 0.5, 1.0, 54321, 1); // Expected: 3 (based on RNG seed)

    // Test random number generation
    float expected_values_12345[] = {0.776939, 0.395173, 0.655770, 0.455296, 0.167368}; // Replace with actual expected values
    float expected_values_54321[] = {0.290433, 0.403477, 0.802450, 0.735965, 0.858177}; // Replace with actual expected values
    test_random_numbers(12345, expected_values_12345);
    test_random_numbers(54321, expected_values_54321);

    return 0;
}
