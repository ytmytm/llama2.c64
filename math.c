// https://raw.githubusercontent.com/drmortalwombat/oscar64/refs/heads/main/include/math.c
// SIN method from https://www.c64-wiki.com/wiki/SIN with optimization for small angles

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
/*
    float s =  6.2831853069 * g;
    float f2 = g*g*g;
    s +=  (-41.341702104)*f2;
    f2 *= g*g;
    s +=  81.605223686*f2;
    f2 *= g*g;
    s +=  (-76.704170257)*f2;
    f2 *= g*g;
    s +=  42.007797122*f2;
    f2 *= g*g;
    s +=  -14.381390672*f2;
*/

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
    
void my_sinpolyeval(float g2, float g, float *r) {
    *r = ((((((-14.381390672) * g2 
        + 42.007797122) * g2 
        - 76.704170257) * g2 
        + 81.605223686) * g2 
        - 41.341702104) * g2 
        + 6.2831853069) * g;
}

void my_sincos(float f, float *s, float *c)
{
    float g = fabs(f);

    // Fast path for very small values
    if (g < 0.5) {
        float f2 = f * f;
        
        // Fifth-order sine approximation: sin(x) ≈ x - x³/6 + x⁵/120
        *s = f * (1.0 - f2 * (1.0/6.0 - f2/120.0));
        
        // Fourth-order cosine approximation: cos(x) ≈ 1 - x²/2 + x⁴/24
        *c = 1.0 - f2 * (0.5 - f2/24.0);
        
        return;
    }
    
    // Regular path for larger values
    float sign_s = f < 0.0 ? -1.0 : 1.0;
    float sign_c = 1.0;
    
    // Normalize to [0, 1) range
    g *= 0.5 / PI;
    g -= floor(g);
    
    // Quadrant handling
    if (g >= 0.5) {
        sign_s = -sign_s;
        sign_c = -sign_c;
        g -= 0.5;
    }
    
    float sine_value, cosine_value;

    if (g >= 0.25) {
        // In this quadrant, sin(x) = cos(π/2 - x) and cos(x) = sin(π/2 - x)
        float g_adjusted = 0.5 - g;
        
        // Calculate cosine using the sine polynomial for the adjusted angle
        float g2 = g_adjusted * g_adjusted;
        my_sinpolyeval(g2, g_adjusted, &cosine_value);
        // For sine, use identity sin(x) = cos(π/2 - x)
        g2 = g * g;
        my_sinpolyeval(g2, g, &sine_value);
    } else {
        // Normal case where x is in first quadrant
        float g2 = g * g;
        my_sinpolyeval(g2, g, &sine_value);      
        // Use identity cos(x) = sin(π/2 - x)
        float g_adjusted = 0.25 - g;
        g2 = g_adjusted * g_adjusted;
        my_sinpolyeval(g2, g_adjusted, &cosine_value);
    }

    *s = sine_value * sign_s;
    *c = cosine_value * sign_c;
}
