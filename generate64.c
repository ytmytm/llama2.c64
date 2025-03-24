/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nnet64.h"
#include "util.h"

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, uint16_t steps) {
    char *empty_prompt = (char*)"";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    uint16_t num_prompt_tokens = 0;

    ui_settopstatus("TOKENIZING...");

    int16_t* prompt_tokens = (int16_t*)malloc((strlen(prompt)+3) * sizeof(int16_t)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        ui_settopstatus("ERROR: NOT TOKENS");
        while (1);
    }
    ui_cleartopstatus();
    ui_setnumberoftokens(num_prompt_tokens);

    ui_gotooutput();

    // prepare nnet buffers
    nnet_init(transformer);

    // start the main loop
    int16_t next;        // will store the next token in the sequence
    int16_t token = prompt_tokens[0]; // kick off with the first token in the prompt
    uint16_t pos = 0;     // position in the sequence
    while (pos < steps) {

        ui_setcurrenttoken(pos+1,steps);

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            ui_settopstatus("sampling");
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        token = next;

    }

    free(prompt_tokens);
}
