
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    int fd;                         // File descriptor for mmap
    void* mmap_ptr;                 // Pointer to mmap'd data
    size_t mmap_size;               // Size of mmap'd data
    char *str_buffer;               // temp buffer for encode (this can be static)
} Tokenizer;

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
void load_tokenizer(Tokenizer* t, const char* load_path);
void save_tokenizer(Tokenizer* t, const char* save_path);

// generate.c
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char* decode(Tokenizer* t, int prev_token, int token);

#endif // TOKENIZER_H
