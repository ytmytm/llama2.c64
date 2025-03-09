/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "nnet.h"
#include "transformer.h"
#include "tokenizer.h"
#include "util.h"


/////////////////////////////
// test/debug nnet.h
void rmsnorm(float* o, float* x, float* weight, int size);
void matmul(float* xout, float* x, float* w, int n, int d);

void dump_matrix(float* xout, int d, const char* name) {
	printf("MATRIX[%i]:%s\n",d,name);
	int i;
	for (i=0;i<d;i++) {
		printf("%f\t",xout[i]);
	}
	printf("\n");
}

void matmul2(float* xout, float* x, float* w, int n, int d) {
 //   printf("matmul xout=%i,xin=%i:wsize=%i\n",4*d,4*n,4*d*n);
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    float *xo = xout;
    float *wi = w;
    float *xi;
    for (uint8_t i = 0; i < d; i++) {
        *xo = 0.0f;
	xi = x;
        for (uint8_t j = 0; j < n; j++) {
	    *xo += (*wi) * (*xi);
	    wi++;
	    xi++;
        }
	xo++;
    }
}


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

    // build the Transformer via the model .bin file
    char *checkpoint_path = "stories260K.bin";  // e.g. out/model.bin
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    printf("build transformer\n");

	// from nnet64.c forward()

	// a few convenience variables
    Config* p = &transformer.config;
    TransformerWeights* w = &transformer.weights;
    RunState* s = &transformer.state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

//    float *x = s->x;
    float *xb = s->xb;
    x[0]=1.0;
    x[1]=0.5;
    x[dim-1]=0.1;
    rmsnorm(xb, x, w->wq, dim);
    dump_matrix(xb, dim, "RMS-XB");

	uint32_t pos = 0; // forward() parameter
	uint32_t l = 0; // loop

	printf("pos=%i\tl=%i\n",pos,l);

        // key and value point to the kv cache
	// XXX offset for REU is *sizeof(float)!
        uint32_t loff;

	loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim; // XXX *sizeof(float)
        s->v = s->value_cache + loff + pos * kv_dim; // XXX *sizeof(float)

	s->xb[0]=1.0;
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0");

        matmul2(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
	dump_matrix(s->q, dim, "[2]SQ-1.0");

	s->xb[0]=1.0;
	s->xb[1]=0.5;
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0-0.5");

        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
	dump_matrix(s->k, dim, "SK");

        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
	dump_matrix(s->v, dim, "SV");

	pos = 5;
	l = 6;

	printf("pos=%i\tl=%i\n",pos,l);

	loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim; // XXX *sizeof(float)
        s->v = s->value_cache + loff + pos * kv_dim; // XXX *sizeof(float)

	s->xb[0]=1.0;
        s->xb[1]=0;
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0");

	s->xb[0]=1.0;
	s->xb[1]=0.5;
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
	dump_matrix(s->q, dim, "SQ-1.0-0.5");

    	return 0;
}

