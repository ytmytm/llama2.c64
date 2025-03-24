// https://raw.githubusercontent.com/drmortalwombat/oscar64/refs/heads/main/include/math.c
// SIN method from https://www.c64-wiki.com/wiki/SIN with optimization for small angles
// EXP method from https://www.c64-wiki.com/wiki/EXP
// general info in https://www.c64-wiki.com/wiki/POLY1

#include <math.h>

float my_sin(float f)
{
	float	g = fabs(f);

    if (g < 0.5) {
        float f2 = f * f;
        // Third-order approximation: sin(x) ≈ x - x³/6
        // return f * (1.0f - f2 / 6.0f);
      
        // Fifth-order approximation for better accuracy while still efficient
        return f * (1.0 - f2 * (1.0/6.0 - f2/120.0));
    }

	float	m = f < 0.0 ? -1.0 : 1.0;

	g *= 0.5 / PI;
	g -= floor(g);

	if (g >= 0.5)
	{
		m = -m;
		g -= 0.5;
	}
	if (g >= 0.25)
		g = 0.5 - g;

    float g2 = g * g;
    // Using Horner's method for polynomial evaluation
    float s = ((((((-14.381390672) * g2 
        + 42.007797122) * g2 
        - 76.704170257) * g2 
        + 81.605223686) * g2 
        - 41.341702104) * g2 
        + 6.2831853069) * g;

	return s * m;
}


float my_cos(float f)
{
	return my_sin(f + 0.5 * PI);
}

float my_exp(float f)
{
    static const union {
        uint32_t i;
        float f;
    } log2e_const = { 0x3FB8AA3B };  // log_2(e) w IEEE 754

//	f *= 1.442695041; // f*=log_2(e)
    f *= log2e_const.f; // f*=log_2(e)

	float	ff = floor(f), g = f - ff; // split into integer and fractional part
	
	int	fi = (int)ff;
	
	union {
		float	f;
		int		i[2];
	}	x;
	x.f = 0;

	x.i[1] = (fi + 0x7f) << 7;
	
    float s = 2.1498763701e-5;
    s = s * g + 1.4352314037e-4;
    s = s * g + 1.3422634825e-3;
    s = s * g + 9.6140170135e-3;
    s = s * g + 5.5505126860e-2;
    s = s * g + 0.24022638460;
    s = s * g + 0.69314718618;
    s = s * g + 1.0;

	return s * x.f;
}

//// 

static inline REUPtr index_uint8(uint8_t a, uint8_t b) {
    return (uint32_t)LUT_OFFSET+(((uint32_t)a << 8) | b) << 1;
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
