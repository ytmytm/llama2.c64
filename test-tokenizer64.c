/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer64.h"
#include "tokenizer64.c"

#include "util.h"
#include "util64.c"

int main(void) {

    // default parameters
    char *tokenizer_path = NULL;  // e.g. out/tokenizer.bin
    float temperature = 1.0;    // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9;           // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string

    // parameter validation/overrides
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    tokenizer_path = (char*)"tokenizer.bin";
    printf("Using processed tokenizer.bin\n");
    load_tokenizer(&tokenizer, tokenizer_path);

    // encode the (string) prompt into tokens sequence
    int16_t num_prompt_tokens = 0;
    int16_t* prompt_tokens;
    int16_t token;
    int16_t next;
    char *piece;

    /* test 1 */

    prompt = (char*)"Once upon a time";

    printf("prompt: %s\n", prompt);
    printf("Allocating generate: prompt_tokens %u bytes\n", (strlen(prompt)+3) * sizeof(int32_t));
    prompt_tokens = (int16_t*)malloc((strlen(prompt)+3) * sizeof(int16_t)); // +3 for '\0', ?BOS, ?EOS
    const int16_t test1_expected[] = { 1, 403, 407, 261, 378 };

    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("prompt_tokens: %u\n", num_prompt_tokens);
    if (num_prompt_tokens != 5) {
        printf("something is wrong, expected 5 prompt tokens\n");
    }
    for (int i = 0; i < 5; i++) {
        printf("%d ", prompt_tokens[i]);
    }
    for (int i = 0; i < 5; i++) {
        if (prompt_tokens[i] != test1_expected[i]) {
            printf("something is wrong, expected prompt_tokens[%d] %d == %d\n", i, prompt_tokens[i], test1_expected[i]);
//            exit(1);
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

    prompt = (char*)"There is a house on a hill. They call it rising sun";

    printf("prompt: %s\n", prompt);
    printf("Allocating generate: prompt_tokens %u bytes\n", (strlen(prompt)+3) * sizeof(int32_t));
    prompt_tokens = (int16_t*)malloc((strlen(prompt)+3) * sizeof(int16_t)); // +3 for '\0', ?BOS, ?EOS
    const int16_t test2_expected[] = { 1, 291, 276, 410, 293, 261, 270, 277, 372, 353, 261, 270, 290, 421, 426, 342, 280, 388, 312, 352, 293, 299, 262, 379  };
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("prompt_tokens: %u\n", num_prompt_tokens);
    if (num_prompt_tokens != 24) {
        printf("something is wrong, expected 24 prompt tokens\n");
    }
    for (int i = 0; i < 24; i++) {
        printf("%d ", prompt_tokens[i]);
    }
    printf("\n");
    for (int i = 0; i < 24; i++) {
        if (prompt_tokens[i] != test2_expected[i]) {
            printf("something is wrong, expected prompt_tokens[%d] %d == %d\n", i, prompt_tokens[i], test2_expected[i]);
//            exit(1);
        }
    }
    printf("\n");

    /* test 3 */
    printf("test3: decode test2\n");
    token = prompt_tokens[0];
    for (int i = 1; i < num_prompt_tokens; i++) {
        next = prompt_tokens[i];
        piece = decode(&tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;
    }
    printf("\n");

    /* test 4 */
    printf("test4: decode expected\n");
    token = test2_expected[0];
    for (int i = 1; i < 24; i++) {
        next = test2_expected[i];
        piece = decode(&tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;
    }
    printf("\n");

    printf("test5: decode directly expected\n");
    for (int i = 0; i < 24; i++) {
//        printf("%u:%ld:[%s]\n", i, test2_expected[i], tokenizer.vocab[test2_expected[i]]);
        printf("%s", tokenizer.vocab[test2_expected[i]]);
    }
    printf("\n");

    printf("test6: decode directly encoded\n");
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%u:%d:[%s]\n", i, prompt_tokens[i], tokenizer.vocab[prompt_tokens[i]]);
//        printf("%s", tokenizer.vocab[prompt_tokens[i]]);
    }
    printf("\n");

    /* test 6 */

    prompt = (char*)"Zoo";

    printf("prompt: %s\n", prompt);
    printf("Allocating generate: prompt_tokens %u bytes\n", (strlen(prompt)+3) * sizeof(int32_t));
    prompt_tokens = (int16_t*)malloc((strlen(prompt)+3) * sizeof(int16_t)); // +3 for '\0', ?BOS, ?EOS
    const int16_t test6_expected[] = { 1, 410, 469, 347  };
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("prompt_tokens: %u\n", num_prompt_tokens);
    if (num_prompt_tokens != 4) {
        printf("something is wrong, expected 4 prompt tokens\n");
    }
    for (uint8_t i = 0; i < 4; i++) {
        printf("%d ", prompt_tokens[i]);
    }
    printf("\n");
    for (uint8_t i = 0; i < 4; i++) {
        if (prompt_tokens[i] != test6_expected[i]) {
            printf("something is wrong, expected prompt_tokens[%d] %d == %d\n", i, prompt_tokens[i], test6_expected[i]);
//            exit(1);
        }
    }
    printf("\n");

    return 0;
}

