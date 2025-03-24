#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#define RANDOM_TESTS 50

#define LUT_SIZE 65536

static uint16_t lut_uint8[LUT_SIZE];

static inline uint32_t index_uint8(uint8_t a, uint8_t b) {
    return ((uint32_t)a << 8) | b;
}

void build_lut_uint8(void) {
    for (uint16_t a = 0; a < 256; ++a) {
        for (uint16_t b = 0; b < 256; ++b) {
            lut_uint8[index_uint8((uint8_t)a, (uint8_t)b)] = (uint16_t)(a * b);
        }
    }
}

uint32_t mult8_uint8_lut(uint8_t a, uint8_t b) {
    return (uint32_t)lut_uint8[index_uint8(a, b)];
}

typedef union {
    float f;
    uint32_t u;
    uint8_t bytes[4];
} FloatBits;

void multiply_float32_via_lut(const float* a, const float* b, float* out_result) {
    if (isnan(*a) || isnan(*b)) {
        *out_result = NAN;
        return;
    }
    if (isinf(*a) || isinf(*b)) {
        *out_result = (*a == 0.0f || *b == 0.0f) ? NAN : copysignf(INFINITY, *a * *b);
        return;
    }
    if (*a == 0.0f || *b == 0.0f) {
        *out_result = 0.0f;
        return;
    }
    if (*a == 1.0f) {
        *out_result = *b;
        return;
    }
    if (*b == 1.0f) {
        *out_result = *a;
        return;
    }

    FloatBits ua = { .f = *a }, ub = { .f = *b };

    uint8_t exp_a = ((ua.bytes[3] & 0x7F) << 1) | (ua.bytes[2] >> 7);
    uint8_t exp_b = ((ub.bytes[3] & 0x7F) << 1) | (ub.bytes[2] >> 7);
    if (exp_a == 0 || exp_b == 0) {
        *out_result = 0.0f;
        return;
    }
    uint8_t result_exp = exp_a + exp_b - 127;

    uint8_t result_sign = (ua.bytes[3] ^ ub.bytes[3]) & 0x80;

    ua.bytes[2] |= 0x80;
    ub.bytes[2] |= 0x80;

    uint8_t* A = &ua.bytes[0];
    uint8_t* B = &ub.bytes[0];

    typedef union {
        uint32_t u[2];  // Dwa 32-bitowe słowa
        uint8_t bytes[8]; // 8 bajtów
    } ResultBytes;

    ResultBytes result_bytes = {0};
    for (uint8_t i = 0; i < 3; ++i) {
        for (uint8_t j = 0; j < 3; ++j) {
            uint16_t partial = mult8_uint8_lut(A[i], B[j]);
            uint8_t lo = partial & 0xFF;
            uint8_t hi = partial >> 8;
            uint8_t pos = i + j;
            uint8_t carry = lo;
            for (uint8_t k = pos; k < 8 && carry > 0; ++k) {
                uint16_t sum = result_bytes.bytes[k] + carry;
                result_bytes.bytes[k] = sum & 0xFF;
                carry = sum >> 8;
            }
            carry = hi;
            for (uint8_t k = pos + 1; k < 8 && carry > 0; ++k) {
                uint16_t sum = result_bytes.bytes[k] + carry;
                result_bytes.bytes[k] = sum & 0xFF;
                carry = sum >> 8;
            }
        }
    }

    FloatBits* result = (FloatBits*)out_result;
    result->u = (result_bytes.u[1] << 9) | (result_bytes.u[0] >> 23);

    if (result->u & (1 << 24)) {
        result->u >>= 1;
        result_exp++;
    }

    result->bytes[2] = (result->bytes[2] & 0x7F) | ((result_exp & 1) << 7);
    result->bytes[3] = (result_sign) | ((result_exp >> 1) & 0x7F);
}

int significant_digit_difference(float a, float b) {
    if (a == b) return 0;

    union { float f; uint32_t u; } ua = { .f = a }, ub = { .f = b };

    if ((ua.u << 1) == 0 || (ub.u << 1) == 0) return 0x800000;

    int32_t exp_a = (ua.u >> 23) & 0xFF;
    int32_t exp_b = (ub.u >> 23) & 0xFF;
    int32_t diff_exp = exp_a - exp_b;

    uint32_t mant_a = (ua.u & 0x7FFFFF) | 0x800000;
    uint32_t mant_b = (ub.u & 0x7FFFFF) | 0x800000;

    if (diff_exp > 0) mant_b >>= diff_exp;
    else if (diff_exp < 0) mant_a >>= -diff_exp;

    return (mant_a > mant_b) ? (mant_a - mant_b) : (mant_b - mant_a);
}

typedef struct { float a, b; } fpair;

int main(void) {
    build_lut_uint8();

    fpair tests[] = {
        {3.14159f, -2.71828f}, {1.0f, 1.0f}, {0.0f, 5.0f}, {5.0f, 0.0f},
        {1.0f, 123.456f}, {123.456f, 1.0f}, {INFINITY, 2.0f}, {-3.0f, INFINITY},
        {NAN, 1.0f}, {0.0f, INFINITY}, {INFINITY, 0.0f},
        {1e-10f, 1e10f}, {1e10f, 1e-10f}, {1e-10f, 1e-10f}, {1e10f, 1e10f},
        {0.9999999f, 1.0000001f}, {-0.5f, -0.5f}, {0.125f, 8.0f}, {3.5f, 0.25f},
        {1.0f / 3.0f, 3.0f}, {1.17549435e-38f, 2.0f}, {1.17549435e-38f, 1.17549435e-38f}
    };

    printf("\nFixed test cases:\n");
    for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); ++i) {
        float a = tests[i].a, b = tests[i].b;
        float res;
        float expected = a * b;
        multiply_float32_via_lut(&a, &b, &res);
        uint32_t mantissa_diff = significant_digit_difference(res, expected);

        printf("% .8e * % .8e = % .8e (expected % .8e, mantissa Δ = %6u)%s\n",
            a, b, res, expected, mantissa_diff, (mantissa_diff <= 4 ? " OK" : " FAIL"));
    }

    printf("\nRandom test cases:\n");
    for (int i = 0; i < RANDOM_TESTS; ++i) {
        float a = (float)rand() / (float)(RAND_MAX / 1e4f) - 5e3f;
        float b = (float)rand() / (float)(RAND_MAX / 1e4f) - 5e3f;
        float res;
        float expected = a * b;
        multiply_float32_via_lut(&a, &b, &res);
        uint32_t mantissa_diff = significant_digit_difference(res, expected);

        printf("% .8e * % .8e = % .8e (expected % .8e, mantissa Δ = %6u)%s\n",
            a, b, res, expected, mantissa_diff, (mantissa_diff <= 4 ? " OK" : " FAIL"));
    }

    return 0;
}
