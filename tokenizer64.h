
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int32_t id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    uint32_t vocab_size;
    uint32_t max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    uint32_t fd;                         // File descriptor for mmap
    char* mmap_ptr;                 // Pointer to mmap'd data
    size_t mmap_size;               // Size of mmap'd data
    char *str_buffer;               // temp buffer for encode (this can be static)
} Tokenizer;

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int32_t vocab_size);
void free_tokenizer(Tokenizer* t);
void load_tokenizer(Tokenizer* t, const char* load_path);
void save_tokenizer(Tokenizer* t, const char* save_path);

// generate.c
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int32_t *tokens, int *n_tokens);
char* decode(Tokenizer* t, int32_t prev_token, int32_t token);

#endif // TOKENIZER_H
