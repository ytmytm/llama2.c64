
/* Inference for Llama-2 Transformer model in pure C */

// C64 port by Maciej 'YTM/Elysium' Witkowiak, 2025

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "tokenizer64.h"

const unsigned char tokenizer_bin[] = {
    #embed "tokenizer2.bin"
};

//////////////////////////////////////////////////////////////////////////////////////////////////////

// https://github.com/gcc-mirror/gcc/blob/master/libiberty/bsearch.c
char*
bsearch (char *key, char **base0,
         size_t nmemb, size_t size,
         int16_t (*compar)(const char *, const char *))
{
	const char *base = (const char *) base0;
	int16_t lim, cmp;
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

int16_t compare_tokens(const char* a, const char* b) {
    return strcmp((char*)a, *(char**)b);
}

/// str_lookup - binary search of sorted vocab
int16_t str_lookup(char *str, Tokenizer *t) { 
    char* res = bsearch(str 
    ,t->sorted_vocab_str
    ,t->vocab_size, sizeof(char*), compare_tokens);
    if (res != NULL) {
        int16_t index = ((uint16_t)(res) - (uint16_t)(t->sorted_vocab_str)) / sizeof(char*);
        return t->sorted_vocab_id[index];
    }
    return -1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

void load_tokenizer(Tokenizer* t) {

    t->mmap_size = sizeof(tokenizer_bin);
    t->mmap_ptr = (char*)tokenizer_bin;

    t->max_token_length = 7;

    // Map the data to Tokenizer structure
    char* ptr = t->mmap_ptr;

    t->vocab_size = *(uint16_t*)ptr; // XXX: get REU byte (?int is 32 or 16?)
    ptr += sizeof(uint16_t);
    t->vocab = (char**)malloc(t->vocab_size * sizeof(char*));

    t->sorted_vocab_str = (char**)malloc(t->vocab_size * sizeof(char*));

    t->vocab_scores = (float*)ptr;
    ptr += t->vocab_size * sizeof(float);
    t->vocab_len = (uint8_t*)ptr;
    ptr += t->vocab_size * sizeof(uint8_t);
    t->sorted_vocab_id = (uint16_t*)ptr;
    ptr += t->vocab_size * sizeof(uint16_t);
    for (uint16_t i=0; i < t->vocab_size; i++) {
        t->vocab[i] = (char*)ptr;
        ptr +=  t->vocab_len[i];
    }
    for (uint16_t i=0; i < t->vocab_size; i++) {
        t->sorted_vocab_str[i] = t->vocab[t->sorted_vocab_id[i]];
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    t->str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));

    for (uint16_t i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

char* decode(Tokenizer* t, int16_t prev_token, int16_t token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {  // XXX test that, didn't work for <0x0a>
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int16_t *tokens, uint16_t *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { 
        exit(1); 
    }

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
        int16_t dummy_prefix = str_lookup((char*)" ", t);
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
        int16_t id = str_lookup(t->str_buffer, t);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int16_t i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)t->str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
//        float best_score = -0.0000000001;
        float best_score = -1000000000;
        int16_t best_id = -1;
        int16_t best_idx = -1;

        for (int16_t i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(t->str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);

            int16_t id = str_lookup(t->str_buffer, t);
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
        for (int16_t i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

}
