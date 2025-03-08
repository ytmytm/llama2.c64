/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transformer64.h"
#include "transformer64.c"

//#include "tokenizer64.h"
//#include "tokenizer64.c"

//#include "util.h"
//#include "util64.c"

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

    Transformer transformer;
    load_transformer(&transformer);
    Config64 *c = transformer.config;

    printf("p->dim=%d\n", c->dim);
    printf("p->hidden_dim=%d\n", c->hidden_dim);
    printf("p->n_layers=%d\n",c->n_layers);    
    printf("p->n_heads=%d\n", c->n_heads);
    printf("p->n_kv_heads=%d\n", c->n_kv_heads);
    printf("p->vocab_size=%d\n", c->vocab_size);
    printf("p->seq_len=%d\n", c->seq_len);
    printf("p->shared_weigths=%d\n", c->shared_weights);

    printf("reu_base FINAL=%lu\n", reu_base);

    return 0;
}

