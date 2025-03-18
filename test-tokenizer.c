/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "tokenizer.h"
#include "util.h"

int main(int argc, char *argv[]) {

    // default parameters
    char *tokenizer_path = NULL;  // e.g. out/tokenizer.bin
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    tokenizer_path = "tokenizer.bin";
    printf("Using processed tokenizer.bin\n");
    load_tokenizer(&tokenizer, tokenizer_path);
    #ifdef DEBUG
    printf("max_token_length: %i\n", tokenizer.max_token_length);
    printf("vocab_size: %i\n", tokenizer.vocab_size);
    printf("tokenizer scores[50]: %f\n", tokenizer.vocab_scores[511]);
    printf("tokenizer vocab[50]: %s\n", tokenizer.vocab[511]);
    printf("tokenizer sorted_vocab[50].str: %s\n", tokenizer.sorted_vocab[511].str);
    printf("tokenizer sorted_vocab[50].id: %i\n", tokenizer.sorted_vocab[511].id);
    #endif

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens;
    int token;
    int next;
    char *piece;

    /* test 1 */

    prompt = "Once upon a time";

    printf("prompt: %s\n", prompt);
    printf("Allocating generate: prompt_tokens %zu bytes\n", (strlen(prompt)+3) * sizeof(int));
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    int test1_expected[] = { 1, 403, 407, 261, 378 };

    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    }
    printf("prompt_tokens: %i\n", num_prompt_tokens);
    if (num_prompt_tokens != 5) {
        printf("something is wrong, expected 5 prompt tokens\n");
    }
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%i ", prompt_tokens[i]);
    }
    for (int i = 0; i < num_prompt_tokens; i++) {
        if (prompt_tokens[i] != test1_expected[i]) {
            printf("something is wrong, expected prompt_tokens[%i] == %i\n", i, test1_expected[i]);
            exit(1);
        }
    }
    printf("\n");

    token = prompt_tokens[0];
    for (int i = 1; i < num_prompt_tokens; i++) {
        next = prompt_tokens[i];
        piece = decode(&tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;
    }
    printf("\n");
    free(prompt_tokens);

    /* test 2 */

    prompt = "There is a house on a hill. They call it rising sun";

    printf("prompt: %s\n", prompt);
    printf("Allocating generate: prompt_tokens %zu bytes\n", (strlen(prompt)+3) * sizeof(int));
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    int test2_expected[] = { 1, 291, 276, 410, 293, 261, 270, 277, 372, 353, 261, 270, 290, 421, 426, 342, 280, 388, 312, 352, 293, 299, 262, 379  };
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("prompt_tokens: %i\n", num_prompt_tokens);
    if (num_prompt_tokens != 24) {
        printf("something is wrong, expected 24 prompt tokens\n");
    }
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%i ", prompt_tokens[i]);
    }
    for (int i = 0; i < num_prompt_tokens; i++) {
        if (prompt_tokens[i] != test2_expected[i]) {
            printf("something is wrong, expected prompt_tokens[%i] == %i\n", i, test2_expected[i]);
            exit(1);
        }
    }
    printf("\n");

    /* test 3 */
    token = prompt_tokens[0];
    for (int i = 1; i < num_prompt_tokens; i++) {
        next = prompt_tokens[i];
        piece = decode(&tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;
    }
    printf("\n");
    free(prompt_tokens);

    /* test 4 */

    prompt = "Zoo";

    printf("prompt: %s\n", prompt);
    printf("Allocating generate: prompt_tokens %zu bytes\n", (strlen(prompt)+3) * sizeof(int));
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    int test4_expected[] = { 1, 410, 469, 347  };
    int test4b_expected[] = { 1, 410, 451, 347  };
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("prompt_tokens: %i\n", num_prompt_tokens);
    if (num_prompt_tokens > 24) {
        printf("something is wrong, expected 24 prompt tokens\n");
    }
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%i ", prompt_tokens[i]);
    }
    for (int i = 0; i < num_prompt_tokens; i++) {
        if (prompt_tokens[i] != test4_expected[i]) {
            printf("something is wrong, expected prompt_tokens[%i] == %i\n", i, test4_expected[i]);
            exit(1);
        }
    }
    printf("\n");

    token = test4b_expected[0];
    printf("test4: decode test4\n");
    for (int i = 1; i < num_prompt_tokens; i++) {
        next = test4b_expected[i];
        piece = decode(&tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;
    }
    printf("\n");
    printf("test4: decode generated test4\n");
    for (int i = 1; i < num_prompt_tokens; i++) {
        next = prompt_tokens[i];
        piece = decode(&tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;
    }
    free(prompt_tokens);


    free_tokenizer(&tokenizer);
    return 0;
}

