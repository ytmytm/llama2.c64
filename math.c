// https://raw.githubusercontent.com/drmortalwombat/oscar64/refs/heads/main/include/math.c

// C64 port by Maciej 'YTM/Elysium' Witkowiak, 2025

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
