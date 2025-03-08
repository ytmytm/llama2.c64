/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transformer64.h"
#include "transformer64.c"

#include "nnet64.h"
#include "nnet64.c"

//#include "tokenizer64.h"
//#include "tokenizer64.c"

//#include "util.h"
//#include "util64.c"

void dump_matrix(float* xout, int d, const char* name) {
	printf("MATRIX:%s,%i\n",name,d);
	int i;
	for (i=0;i<d;i++) {
		printf("%f",xout[i]);
	}
}

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

	// from nnet64.c forward()

	// a few convenience variables
    Config64* p = transformer.config;
    TransformerWeights64* w = &transformer.weights;
    RunState64* s = &transformer.state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

	uint32_t pos = 0; // forward() parameter
	uint32_t l = 0; // loop

        // key and value point to the kv cache
	// XXX offset for REU is *sizeof(float)!
        uint32_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim; // XXX *sizeof(float)
        s->v = s->value_cache + loff + pos * kv_dim; // XXX *sizeof(float)

        // qkv matmuls for this position
	// s->q is local!
	// s->xb too
	// s->k, s->v not
	// w-> is only remote
	// w-> ... '+' means *sizeof(float)
//        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
//        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
//        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);


	// dump xout (first+last parameter of matmul)
	dump_matrix(s->q, dim, "SQ");

    return 0;
}

