/* Inference for Llama-2 Transformer model in pure C */

// C64 port by Maciej 'YTM/Elysium' Witkowiak, 2025

#include <stdio.h>
#include <ctype.h>

// ----------------------------------------------------------------------------

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

    for (uint8_t i = 0; i < strlen(piece); i++) {
        char c = piece[i];
        // Convert ASCII to PETSCII
        if (c >= 0x41 && c <= 0x5A) {
            c += 0x80; // Convert uppercase ASCII to PETSCII
        } else if (c >= 0x61 && c <= 0x7A) {
            c -= 0x20; // Convert lowercase ASCII to PETSCII
        }
        putpch(c);
    }
}
