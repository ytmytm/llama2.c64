
/* Inference for Llama-2 Transformer model in pure C */

// C64 port by Maciej 'YTM/Elysium' Witkowiak, 2025

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char** vocab;
    float* vocab_scores;
    char** sorted_vocab_str;
    uint8_t* vocab_len;
    uint16_t* sorted_vocab_id;
    uint16_t vocab_size;
    uint8_t max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    char* mmap_ptr;                 // Pointer to mmap'd data
    size_t mmap_size;               // Size of mmap'd data
    char *str_buffer;               // temp buffer for encode (this can be static)
} Tokenizer;

void load_tokenizer(Tokenizer* t);

// generate.c
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int16_t *tokens, uint16_t *n_tokens);
char* decode(Tokenizer* t, int16_t prev_token, int16_t token);

#endif // TOKENIZER_H
