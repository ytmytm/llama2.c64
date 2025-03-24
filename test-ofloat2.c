
#include <stdint.h>
//#include <math.h>
#include <string.h>
//#include <stdbool.h>
//#include <stdlib.h>
#include <stdio.h>

#define RANDOM_TESTS 50

typedef uint32_t REUPtr;

// ----------------------------------------------------------------------------
// REU functions (access to transformer weights)

struct REU
{
    volatile uint8_t status;
    volatile uint8_t command;
    volatile uint16_t c64_base;
    volatile uint16_t reu_base;
    volatile uint8_t reu_base_bank;
    volatile uint16_t length;
    volatile uint8_t irq;
    volatile uint8_t control;
};

#define reu     (*((struct REU *)0xdf00))

void REU_init() {
    reu.control = 0; // increment both addresses
}

void REU_getf(REUPtr ptr, volatile float* out, uint16_t size) {
    reu.c64_base = (uint16_t)out;
    reu.reu_base = (uint16_t)(ptr & 0xFFFF);
    reu.reu_base_bank = (uint8_t)((ptr >> 16) & 0xFF);
    reu.length = size;
    reu.command = 0x91; // read from REU, execute immediately
}

void REU_putf(REUPtr ptr, volatile float* in, uint16_t size) {
    reu.c64_base = (uint16_t)in;
    reu.reu_base = (uint16_t)(ptr & 0xFFFF);
    reu.reu_base_bank = (uint8_t)((ptr >> 16) & 0xFF);
    reu.length = size;
    reu.command = 0x90; // write to REU, execute immediately
}

static inline REUPtr index_uint8(uint8_t a, uint8_t b) {
    return (((uint32_t)a << 8) | b) << 1;
}

void build_lut_uint8(void) {
    REUPtr index;
    uint16_t result;
    for (uint16_t a = 0; a < 256; ++a) {
        for (uint16_t b = 0; b < 256; ++b) {
            index = index_uint8((uint8_t)a, (uint8_t)b);
            result = a*b;
            REU_putf(index, (float*)&result, sizeof(uint16_t));
        }
    }
}

uint16_t mult8_uint8_lut(uint8_t a, uint8_t b) {
    REUPtr index = index_uint8(a, b);
    uint16_t result;
    REU_getf(index, (float*)&result, sizeof(uint16_t));
    return result;
}

typedef union {
    float f;
    uint32_t u;
    uint8_t bytes[4];
} FloatBits;


typedef union {
    uint32_t u[2];  // Dwa 32-bitowe słowa
    uint8_t bytes[8]; // 8 bajtów
} ResultBytes;


void multiply_float32_via_lut(const float* a, const float* b, float* out_result) {
//    if (isnan(*a) || isnan(*b)) {
//        *out_result = NAN;
//        return;
//    }
//    if (isinf(*a) || isinf(*b)) {
//        *out_result = (*a == 0.0f || *b == 0.0f) ? NAN : copysignf(INFINITY, *a * *b);
//        return;
//    }
    if (*a == 0.0 || *b == 0.0) {
        *out_result = 0.0;
        return;
    }
    if (*a == 1.0) {
        *out_result = *b;
        return;
    }
    if (*b == 1.0) {
        *out_result = *a;
        return;
    }

    FloatBits ua = { .f = *a }, ub = { .f = *b };

    uint8_t exp_a = ((ua.bytes[3] & 0x7F) << 1) | (ua.bytes[2] >> 7);
    uint8_t exp_b = ((ub.bytes[3] & 0x7F) << 1) | (ub.bytes[2] >> 7);
    if (exp_a == 0 || exp_b == 0) {
        *out_result = 0.0;
        return;
    }
    uint8_t result_exp = exp_a + exp_b - 127;

    uint8_t result_sign = (ua.bytes[3] ^ ub.bytes[3]) & 0x80;

    ua.bytes[2] |= 0x80;
    ub.bytes[2] |= 0x80;

    uint8_t* A = &ua.bytes[0];
    uint8_t* B = &ub.bytes[0];

    ResultBytes result_bytes;
    memset(&result_bytes, 0, sizeof(ResultBytes));
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

    result->bytes[0] = (result_bytes.bytes[2] >> 7) | (result_bytes.bytes[3] << 1);
    result->bytes[1] = (result_bytes.bytes[3] >> 7) | (result_bytes.bytes[4] << 1);
    result->bytes[2] = (result_bytes.bytes[4] >> 7) | (result_bytes.bytes[5] << 1);
    result->bytes[3] = (result_bytes.bytes[5] >> 7) | (result_bytes.bytes[6] << 1);

    if (result->bytes[3] & 0x01) {
        result->u >>= 1;
        result_exp++;
    }

    result->bytes[2] = (result->bytes[2] & 0x7F) | ((result_exp & 1) << 7);
    result->bytes[3] = (result_sign) | ((result_exp >> 1) & 0x7F);

}

typedef struct { float a, b; } fpair;

int main(void) {
    printf("Generating LUT...\n");
    build_lut_uint8();

    fpair tests[] = {
        {3.14159, -2.71828}, {1.0, 1.0}, {0.0, 5.0}, {5.0, 0.0},
        {1.0, 123.456}, {123.456, 1.0},
        {0.9999999, 1.0000001}, {-0.5, -0.5}, {0.125, 8.0}, {3.5, 0.25},
        {1.0 / 3.0, 3.0},
    };

    printf("\nFixed test cases:\n");
    for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); ++i) {
        float a = tests[i].a, b = tests[i].b;
        float res;
        float expected = a * b;
        multiply_float32_via_lut(&a, &b, &res);

        printf("%f * %f = %f (expected %f)\n", a, b, res, expected);
    }

    printf("\nRandom test cases:\n");
    FloatBits a, b;
    for (int i = 0; i < RANDOM_TESTS; ++i) {
        a.u = rand();
        b.u = rand();
        float res;
        float expected = a.f * b.f;
        multiply_float32_via_lut(&a.f, &b.f, &res);

        printf("%f * %f = %f (expected %f)\n", a.f, b.f, res, expected);
    }

    return 0;
}
