
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "tokenizer64.h"

const unsigned char tokenizer_bin[] = {
    #embed "tokenizer.bin"
};

// https://github.com/gcc-mirror/gcc/blob/master/libiberty/bsearch.c
char*
bsearch (const void *key, const void *base0,
         size_t nmemb, size_t size,
         int32_t (*compar)(const void *, const void *))
{
	const char *base = (const char *) base0;
	int lim, cmp;
	const char *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (char *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return (NULL);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

// needed for qsorting the vocabulary and bsearch
int32_t compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str); // XXX pointers point to REU memory
}

void load_tokenizer(Tokenizer* t, const char* load_path) {

    t->mmap_size = 8791; // tokenizer.bin filesize
    t->mmap_size = sizeof(tokenizer_bin);
//    t->mmap_ptr = (void*)0; // REU base address
    t->mmap_ptr = (char*)tokenizer_bin;

    // Map the data to Tokenizer structure
    char* ptr = t->mmap_ptr;
    printf("ptr: %lu\n", ptr);
    t->max_token_length = *(uint32_t*)ptr; // XXX: get REU byte (?int is 32 or 16?)
    ptr += sizeof(uint32_t);
    t->vocab_size = *(uint32_t*)ptr; // XXX: get REU byte (?int is 32 or 16?)
    ptr += sizeof(uint32_t);
    printf("ptr: %lu\n", ptr);

    printf("max_token_length: %lu\n", t->max_token_length);
    printf("vocab: %lu\n", t->vocab_size);

    printf("Allocating vocab: %lu bytes\n", t->vocab_size * sizeof(char*));
    t->vocab = (char**)malloc(t->vocab_size * sizeof(char*));

    printf("Allocating vocab_scores: %lu bytes\n", t->vocab_size * sizeof(float));
    t->vocab_scores = (float*)malloc(t->vocab_size * sizeof(float));

    printf("Allocating sorted_vocab: %lu bytes\n", t->vocab_size * sizeof(TokenIndex));
    t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));

    printf("ptr: %x\n", ptr);
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        t->vocab_scores[i] = *(float*)ptr; // XXX: get REU byte (?float is 32 or 16?)
//        printf("%f:%f\t", t->vocab_scores[i],*(float*)ptr);
        ptr += sizeof(float);

        uint32_t len = *(uint32_t*)ptr; // XXX: get REU byte (?int is 32 or 16?)
//        printf("%lu:%lu\t", len, *(uint32_t*)ptr);
        ptr += sizeof(uint32_t);

        t->vocab[i] = (char*)ptr; // XXX: get REU byte (?char* is 32 or 16?)
//        printf("[%s]\n", t->vocab[i]);
        ptr += len;
    }

    for (int i = 0; i < t->vocab_size; i++) {
        int32_t id = *(uint32_t*)ptr; // XXX: get REU byte (?int is 32 or 16?)
        ptr += sizeof(int32_t);
        t->sorted_vocab[i].str = t->vocab[id];
        t->sorted_vocab[i].id = id;
    }
    printf("Loaded %lu bytes of tokenizer data [%lu]\n", (uint32_t)ptr-(uint32_t)(t->mmap_ptr), t->mmap_size);

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    printf("Allocating str_buffer: %lu bytes\n", (t->max_token_length*2 +1 +2) * sizeof(char));
    t->str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////

char* decode(Tokenizer* t, int32_t prev_token, int32_t token) {
    char *piece = t->vocab[token]; // XXX copy that string from REU into buffer
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

int32_t str_lookup(char *str, TokenIndex *sorted_vocab, uint32_t vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int32_t *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { printf("cannot encode NULL text\n"); exit(1); }

    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int32_t dummy_prefix = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        t->str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        t->str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        uint32_t id = str_lookup(t->str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)t->str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -0.0000000001;
        int32_t best_id = -1;
        int32_t best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(t->str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int32_t id = str_lookup(t->str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

}
