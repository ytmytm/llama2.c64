/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transformer64.h"
#include "transformer64.c"

#include "math.c"

#include "nnet64.h"
#include "nnet64.c"

//#include "tokenizer64.h"
//#include "tokenizer64.c"

//#include "util.h"
//#include "util64.c"

#include <c64/memmap.h>

#pragma region( main, 0x0a00, 0xd000, , , {code, data, bss, heap, stack} )

//#pragma stacksize(4096)
//#pragma heapsize(8192)

void dump_matrix(REUPtr xout, int d, const char* name) {
	printf("MATRIX:%s,%d\n",name,d);
	int i;
    float f;
	for (i=0;i<d;i++) {
        REU_getf(xout, &f, sizeof(float));
		printf("%f\t",f);
        xout += sizeof(float);
	}
	printf("\n");
}

void dump_matrix_local(float* xout, int d, const char* name) {
	printf("MATRIX:%s,%d\n",name,d);
	int i;
	for (i=0;i<d;i++) {
		printf("%f\t",xout[i]);
	}
	printf("\n");
}

int main(void) {

    mmap_set(MMAP_NO_BASIC);

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

    printf("REUBASE FINAL=%lu\n", reu_base);

	// from nnet64.c forward()
    Config64* p = transformer.config;
    TransformerWeights64* w = &transformer.weights;
    RunState64* s = &transformer.state;
    int dim = p->dim;
if (0) {
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
        s->k = s->key_cache + (loff + pos * kv_dim)*sizeof(float);
        s->v = s->value_cache + (loff + pos * kv_dim)*sizeof(float);

        // qkv matmuls for this position
	// s->xb is local!
	// s->q, s->k, s->v not
	// w-> is only remote
	// w-> ... '+' means *sizeof(float)
//        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
//        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
//        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

    printf("POS=%d\tL=%d\n",pos,l);
    s->xb[0]=1.0;
    matmul(s->q, s->xb, w->wq + (l*dim*dim)*sizeof(float), dim, dim);
	// dump xout (first+last parameter of matmul)
	dump_matrix(s->q, dim, "SQ");

	s->xb[0]=1.0;
	s->xb[1]=0.5;
    matmul(s->q, s->xb, w->wq + (l*dim*dim)*sizeof(float), dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0-0.5");

    matmul(s->k, s->xb, w->wk + (l*dim*kv_dim)*sizeof(float), dim, kv_dim);
	dump_matrix(s->k, dim, "SK");

    matmul(s->v, s->xb, w->wv + (l*dim*kv_dim)*sizeof(float), dim, kv_dim);
	dump_matrix(s->v, dim, "SV");

	pos = 5;
	l = 6;

	printf("POS=%d\tL=%d\n",pos,l);

	loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + (loff + pos * kv_dim)*sizeof(float); // XXX *sizeof(float)
    s->v = s->value_cache + (loff + pos * kv_dim)*sizeof(float); // XXX *sizeof(float)

	s->xb[0]=1.0;
    s->xb[1]=0;
    matmul(s->q, s->xb, w->wq + (l*dim*dim)*sizeof(float), dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0");

	s->xb[0]=1.0;
    s->xb[1]=0.5;
    matmul(s->q, s->xb, w->wq + (l*dim*dim)*sizeof(float), dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0-0.5");

    float *xb = s->xb;
    x[0]=1.0;
    x[1]=0.5;
    x[dim-1]=0.1;
    rmsnorm(xb, x, w->wq, dim);
    dump_matrix_local(xb, dim, "RMS-XB");

    softmax(w->wq, dim);
    dump_matrix(w->wq, dim, "SOFTMAX-XB");
}

    float *logits;

    // eksperyment
//    logits = forward(&transformer, 410, 1);
//    dump_matrix_local(logits, 64, "LOGITS");
    // pierwsza runda wyglada, ok dopiero druga jest zla(?)
    // liczenie 410 na pozycji 1 (najpierw) ok
    // potem 1 na pozycji 0 (potem) też ok
    // potem 410 na pozycji 1 (znowu) RESID2-X jest zupełnie inne; ATTSOFTMAX też, już na layer=0
    // na debug skrócić layers do 2?
    // to coś, co zależy od pos?

    // Zoo
    // tokens from 'Zoo'
    // logits[1] should be 8.404342, then 1.329065, then 0.889831
    //                     8.404346       1.329061       2.317674(!)
    // on 3rd token: RMS-X-FINAL is 1.107986        0.320713        -1.164740
    //                should be     1.476869        1.410065        -1.209479

    REUPtr content_row = w->token_embedding_table + ((uint32_t)469 * dim)*sizeof(float);
    logits = s->x;
    REU_getf(content_row, logits, dim*sizeof(float));
    dump_matrix_local(logits, dim, "TOKEN469");

    logits = forward(&transformer, 1, 0);
//    dump_matrix_local(logits, p->vocab_size, "LOGITS");
    dump_matrix_local(logits, 64, "LOGITS");
    
    logits = forward(&transformer, 410, 1);
    dump_matrix_local(logits, 64, "LOGITS");

    logits = forward(&transformer, 469, 2);
    dump_matrix_local(logits, 64, "LOGITS");

//    REUPtr content_row = w->token_embedding_table + ((uint32_t)469 * dim)*sizeof(float);
//    REU_getf(content_row, logits, dim*sizeof(float));
//    dump_matrix_local(logits, dim, "TOKEN469");

    while (true);

    return 0;
}

