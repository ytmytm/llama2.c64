/* Inference for Llama-2 Transformer model in pure C */

#ifndef UTIL_H
#define UTIL_H

// ----------------------------------------------------------------------------
// utilities: time

// return time in milliseconds, for benchmarking the model speed
long time_in_ms();

// for generate only
void safe_printf(char *piece);

#endif  // UTIL_H