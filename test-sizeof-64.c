#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// int = 2 bytes
// long = 4 bytes
// any pointer is 2 bytes (can't be used for REU)
// void* is constant, can't be changed

const unsigned char tokenizer_bin[] = {
    #embed "tokenizer.bin"
};

typedef struct {
    char** vocab;
    float* vocab_scores;
//    TokenIndex *sorted_vocab;
    int32_t vocab_size;
    int32_t max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    int32_t fd;                         // File descriptor for mmap
    void* mmap_ptr;                 // Pointer to mmap'd data
    size_t mmap_size;               // Size of mmap'd data
    char *str_buffer;               // temp buffer for encode (this can be static)
} Tokenizer;

Tokenizer tt;

Tokenizer* t;

int main(void) {

    printf("sizeof(char): %u\n", sizeof(char)); // 1
    printf("sizeof(short): %u\n", sizeof(short)); // 2
    printf("sizeof(int): %u\n", sizeof(int));   // 2
    printf("sizeof(long): %u\n", sizeof(long)); // 4
    printf("sizeof(long int): %u\n", sizeof(long int)); // 4
    printf("sizeof(float): %u\n", sizeof(float)); // 4
    printf("sizeof(void*): %u\n", sizeof(void*)); // 2
    printf("sizeof(void**): %u\n", sizeof(void**)); // 2
    printf("sizeof(int8_t): %u\n", sizeof(int8_t)); // 1
    printf("sizeof(int16_t): %u\n", sizeof(int16_t)); // 2
    printf("sizeof(int32_t): %u\n", sizeof(int32_t)); // 4
    printf("sizeof(size_t): %u\n", sizeof(size_t)); // 2
    printf("\n");

    for (uint8_t i=0; i<8; i++) {
        printf("%d\n", tokenizer_bin[i]);
    }

    t = &tt;
    t->mmap_ptr = tokenizer_bin;

    char* ptr = t->mmap_ptr;

//    long vocab = *(int32_t*)&tokenizer_bin[4];
//    long max_token_length = *(int32_t*)&ptr[0];

//    printf("max_token_length: %lu\n", max_token_length);
//    printf("vocab: %lu\n", vocab);

    printf("ptr: %u\n", ptr);
    t->max_token_length = *(int32_t*)ptr;
    ptr += sizeof(int32_t);
//    ptr = &ptr[4];
    printf("ptr: %lu\n", ptr);
    t->vocab_size = *(int32_t*)ptr; // XXX: get REU byte (?int is 32 or 16?)
  //  t->vocab_size = *(int32_t*)&ptr[4]; // XXX: get REU byte (?int is 32 or 16?)
    ptr += sizeof(int32_t);
    printf("ptr: %lu\n", ptr);

    printf("max_token_length: %lu\n", t->max_token_length);
    printf("vocab: %lu\n", t->vocab_size);

    printf("Allocating vocab: %lu bytes\n", t->vocab_size * sizeof(char*));
    t->vocab = (char**)malloc(t->vocab_size * sizeof(char*));

    printf("\n");
    return 0;
}

