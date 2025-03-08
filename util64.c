/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <time.h>
#include <ctype.h>

// ----------------------------------------------------------------------------
// utilities: time


// needed for generate only
void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    if (piece[0] < 0) { printf("\n"); return; } // for mmaped tokenizer only, why?
    printf("%s", piece);
}
