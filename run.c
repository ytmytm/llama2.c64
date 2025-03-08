/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "tokenizer.h"
#include "sampler.h"
#include "nnet.h"
#include "transformer.h"
#include "util.h"
#include "generate.h"

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = NULL;  // e.g. out/tokenizer.bin
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    printf("build transformer\n");

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    printf("Vocab size: %i\n",transformer.config.vocab_size);
    if (tokenizer_path == NULL) {
        tokenizer_path = "tokenizer.bin";
        printf("Using processed tokenizer.bin\n");
        load_tokenizer(&tokenizer, tokenizer_path);
    } else {
        build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
        save_tokenizer(&tokenizer, "tokenizer.bin");
        printf("Saved processed tokenizer.bin\n");
    }
    #ifdef DEBUG
    printf("max_token_length: %i\n", tokenizer.max_token_length);
    printf("vocab_size: %i\n", tokenizer.vocab_size);
    printf("tokenizer scores[50]: %f\n", tokenizer.vocab_scores[511]);
    printf("tokenizer vocab[50]: %s\n", tokenizer.vocab[511]);
    printf("tokenizer sorted_vocab[50].str: %s\n", tokenizer.sorted_vocab[511].str);
    printf("tokenizer sorted_vocab[50].id: %i\n", tokenizer.sorted_vocab[511].id);
    #endif

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    printf("build sampler\n");

    // run!
    printf("running...\n");
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
