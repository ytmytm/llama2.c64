/* Inference for Llama-2 Transformer model in pure C */

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "nnet64.h"
//#include "nnet.h"

void dump_matrix(REUPtr xout, int d, const char* name);
void dump_matrix_local(float* xout, int d, const char* name);

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, REUPtr weight, uint8_t size) {
    #ifdef DEBUG
    printf("RMSNORM :SIZE=%d\n",size);
    #endif
    float wif;
    REUPtr wi = weight;
    // calculate sum of squares
    float ss = 0.0;
    for (uint8_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 0.00001;
    ss = 1.0 / sqrt(ss);
    // normalize and scale
    for (uint8_t j = 0; j < size; j++) {
        REU_getf(wi, &wif, sizeof(float)); // XXX: can be faster if whole row is read once into weights[size] then use weights[j] instead of wif
        wi += sizeof(float);
        o[j] = wif * (ss * x[j]);
    }
}

// x is remote, size is sampler->vocab_size (uint_16t)
void softmax(REUPtr x, uint16_t size) {
    #ifdef DEBUG
    printf("SOFTMAX SIZE=%d\n",size);
    #endif
    float xif;
    REUPtr xi;
    // XXX64 would be faster with local buffer for x[size] to operate here and write back at the end
    // find max value (for numerical stability)
    float max_val;
    xi = x;
    REU_getf(xi, &max_val, sizeof(float)); // x[0]
    xi += sizeof(float); // loop below starts with x[1]
    for (uint16_t i = 1; i < size; i++) {
        REU_getf(xi, &xif, sizeof(float));
        if (xif > max_val) {
            max_val = xif;
        }
        xi += sizeof(float);
    }
    // exp and sum
    float sum = 0.0;
    xi = x;
    for (uint16_t i = 0; i < size; i++) {
        REU_getf(xi, &xif, sizeof(float));
        xif = exp(xif - max_val);
        REU_putf(xi, &xif, sizeof(float)); // write back locally changed
        sum += xif;
        xi += sizeof(float);
    }
    // normalize
    xi = x;
    for (uint16_t i = 0; i < size; i++) {
        REU_getf(xi, &xif, sizeof(float));
        xif /= sum;
        REU_putf(xi, &xif, sizeof(float)); // write back locally changed
        xi += sizeof(float);
    }
}

// xout is remote, x is local, w is remote, n/d are always dim
void matmul(REUPtr xout, float* x, REUPtr w, uint8_t n, uint8_t d) {
    #ifdef DEBUG
    printf("MATMUL N=%d,D=%d\n",n,d);
    #endif
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    REUPtr xo = xout;
    float xof;
    REUPtr wi = w;
    float wif;
    float *xi;
    for (uint8_t i = 0; i < d; i++) {
        xof = 0.0;
        xi = x;
        for (uint8_t j = 0; j < n; j++) {
            REU_getf(wi, &wif, sizeof(float)); // XXX: can be faster if whole row is read once
            xof += wif * (*xi);
//            printf("[%d,%d],[%f]*[%f]=%f\n",i,j,wif,*xi,xof);
            wi += sizeof(float);
            xi++;
        }
        REU_putf(xo, &xof, sizeof(float)); // XXX: can be faster if whole row is written once
        xo += sizeof(float);
    }
}

// xout is local, x is local, w is remote, n/d are always dim
void matmul_l(float* xout, float* x, REUPtr w, uint8_t n, uint8_t d) {
    #ifdef DEBUG
    printf("MATMUL-L DIMS N=%d,D=%d\n",n,d);
    #endif
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    float *xo = xout;
    REUPtr wi = w;
    float wif;
    float *xi;
    for (uint8_t i = 0; i < d; i++) {
        *xo = 0.0;
        xi = x;
        for (uint8_t j = 0; j < n; j++) {
            REU_getf(wi, &wif, sizeof(float)); // XXX: can be faster if whole row is read once
            *xo += wif * (*xi);
//            printf("[%d,%d],[%f]*[%f]=%f\n",i,j,wif,*xi,xof);
            wi += sizeof(float);
            xi++;
        }
        xo++;
    }
}

// xout is local, x is local, w is remote, n/d are always dim
void matmul_ll(float* xout, float* x, REUPtr w, uint8_t n, uint16_t d) {
    #ifdef DEBUG
    printf("MATMUL-LL DIMS N=%d,D=%d\n",n,d);
    #endif
    //    printf("MATMUL-L XOUT=%d,XIN=%d:WSIZE=%d\n",4*d,4*n,(uint16_t)4*d*n);
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        float *xo = xout;
        REUPtr wi = w;
        float wif;
        float *xi;
        for (uint16_t i = 0; i < d; i++) {
            *xo = 0.0;
            xi = x;
            for (uint8_t j = 0; j < n; j++) {
                REU_getf(wi, &wif, sizeof(float)); // XXX: can be faster if whole row is read once
                *xo += wif * (*xi);
    //            printf("[%d,%d],[%f]*[%f]=%f\n",i,j,wif,*xi,xof);
                wi += sizeof(float);
                xi++;
            }
            xo++;
        }
    }
    
void rope(uint8_t dim, RunState64 *s, uint8_t head_size, uint16_t pos, uint8_t kv_dim)
{
    #ifdef DEBUG
    printf("ROPE: %d:%d\n", dim, kv_dim);
    #endif
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    REUPtr vecq = s->q; // the vector to rotate (query or key)
    REUPtr veck = s->k; // the vector to rotate (query or key)
    float vi[2];
    float vo[2];
    for (uint8_t i = 0; i < dim; i += 2)
    {
        int head_dim = i % head_size;
//        printf("%d:HEAD_SIZE=%d,HEAD_DIM=%d\n",i,head_size,head_dim);
        float val = pos * 1.0 / pow(10000.0, head_dim / (float)head_size);
        float fcr = cos(val);
        float fci = sin(val);
//        printf("%d:VAL=%f,FCR=%f,FCI=%f\n",i,val,fcr,fci);
        uint8_t rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (uint8_t v = 0; v < rotn; v++)
        {
            REUPtr vec = v == 0 ? vecq : veck; // the vector to rotate (query or key)
            REU_getf(vec, &vi[0], 2 * sizeof(float));
//            printf("%d:%d:%f-%f\t",i,v,vi[0],vi[1]);
            vo[0] = vi[0] * fcr - vi[1] * fci;
            vo[1] = vi[0] * fci + vi[1] * fcr;
            REU_putf(vec, &vo[0], 2 * sizeof(float));
//            printf("%f-%f\n",vo[0],vo[1]);
            //                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key) // XXX64: all of these are remote
            //                float v0 = vec[i];
            //                float v1 = vec[i+1];
            //                vec[i]   = v0 * fcr - v1 * fci;
            //                vec[i+1] = v0 * fci + v1 * fcr;
        }
        vecq += 2 * sizeof(float);
        veck += 2 * sizeof(float);
    }
}

void attn(Config64 *p, RunState64 *s, uint8_t head_size, uint16_t pos, uint32_t loff, uint8_t kv_dim, uint8_t kv_mul)
{
    #ifdef DEBUG
    printf("ATTN: %d,%d\n", p->n_heads,head_size);
    #endif
    // multihead attention. iterate over all heads
    for (uint8_t h = 0; h < p->n_heads; h++)
    {
        // get the query vector for this head
        REUPtr q = s->q + ((uint32_t)h * head_size) * sizeof(float); // XXX64: q is remote
//            float* q = s->q + h * head_size; // XXX64: q is remote
        // attention scores for this head
        REUPtr att = s->att + ((uint32_t)h * p->seq_len) * sizeof(float); // XXX64: att is remote
//            float* att = s->att + h * p->seq_len; // XXX64: att is remote
        // iterate over all timesteps, including the current one
        REUPtr atti = att;
        for (uint16_t t = 0; t <= pos; t++)
        {
            // get the key vector for this head and at this timestep
            REUPtr qq = q;
            REUPtr k = s->key_cache + ((uint32_t)loff + t * kv_dim + (h / kv_mul) * head_size) * sizeof(float); // XXX64: key_cache is remote
//                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size; // XXX64: key_cache is remote
            // calculate the attention score as the dot product of q and k
            float score = 0.0;
            float qi, ki;
            for (uint8_t i = 0; i < head_size; i++)
            {
                REU_getf(qq, &qi, sizeof(float)); // XXX can buffer whole thing
                REU_getf(k, &ki, sizeof(float));
                score += qi * ki;
//                printf("%d:%f:%f,SCORE=%f\n",i,qi,ki,score);
                qq += sizeof(float);
                k += sizeof(float);
//                    score += q[i] * k[i];
            }
//            printf("%d:%d,SCORE=%f\n",h,t,score);
            score /= sqrt(head_size);
            // save the score to the attention buffer
            REU_putf(atti, &score, sizeof(float)); // XXX buffer
            atti += sizeof(float);
//                att[t] = score;
        }
//        dump_matrix(att, pos, "ATT");

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);
//        dump_matrix(att, pos, "ATTSOFTMAX");

        // weighted sum of the values, store back into xb
        float *xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        atti = att;
        for (uint16_t t = 0; t <= pos; t++)
        {
            // get the value vector for this head and at this timestep
            REUPtr v = s->value_cache + ((uint32_t)loff + t * kv_dim + (h / kv_mul) * head_size) * sizeof(float);
//                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a;
            REU_getf(atti, &a, sizeof(float));
            atti += sizeof(float);
//                float a = att[t];
            float vi;
            // accumulate the weighted value into xb
            for (uint8_t i = 0; i < head_size; i++)
            {
                REU_getf(v, &vi, sizeof(float));
                xb[i] += a * vi;
                v += sizeof(float);
//                    xb[i] += a * v[i];
            }
        }
    }
}

// assumption: n_heads, dim, hidden_dim are <256
float* forward(Transformer* transformer, uint16_t token, uint16_t pos) {

    // a few convenience variables
    Config64* p = transformer->config;
    TransformerWeights64* w = &transformer->weights; // XXX64:all are remote
    RunState64* s = &transformer->state;
    float *x = s->x; // XXX64: x, s->x local
    // XXX64: some (all) of these could be uint16_t (like all config values)
    uint8_t dim = p->dim;
    uint8_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    uint8_t kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    uint8_t hidden_dim =  p->hidden_dim;
    uint8_t head_size = dim / p->n_heads;

    // copy the token embedding into x
    // XXX64: token_embedding_table is remote, x is local
//    float* content_row = w->token_embedding_table + token * dim;
//    memcpy(x, content_row, dim*sizeof(*x));
    REUPtr content_row = w->token_embedding_table + ((uint32_t)token * dim)*sizeof(float);
    REU_getf(content_row, x, dim*sizeof(float));
//    for (uint16_t i = 0; i < dim; i++) {
//        REU_getf(content_row, &x[i], sizeof(float));
//        content_row += sizeof(float);
//    }

//    dump_matrix_local(x, dim, "TOKEN");

    // forward all the layers
    for(uint8_t l = 0; l < p->n_layers; l++) {
        #ifndef DEBUG
        printf("LAYER: %d OF %d\n",l,p->n_layers);
        #endif

        // attention rmsnorm
        // XXX64: xb is local, x is local, weight is remote
        rmsnorm(s->xb, x, w->rms_att_weight + ((uint32_t)l*dim)*sizeof(float), dim);
//        dump_matrix_local(s->xb, dim, "RMSNORM");

        // key and value point to the kv cache
        uint32_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + (loff + pos * kv_dim)*sizeof(float);
        s->v = s->value_cache + (loff + pos * kv_dim)*sizeof(float);

        // qkv matmuls for this position
//        dump_matrix(w->wq + ((uint32_t)l*dim*dim)*sizeof(float), dim, "WQ");
        matmul(s->q, s->xb, w->wq + ((uint32_t)l*dim*dim)*sizeof(float), dim, dim);
//        dump_matrix(s->q, dim, "SQ");
        matmul(s->k, s->xb, w->wk + ((uint32_t)l*dim*kv_dim)*sizeof(float), dim, kv_dim);
//        dump_matrix(s->k, kv_dim, "SK");
        matmul(s->v, s->xb, w->wv + ((uint32_t)l*dim*kv_dim)*sizeof(float), dim, kv_dim);
//        dump_matrix(s->v, kv_dim, "SV");

        rope(dim, s, head_size, pos, kv_dim); // modifies s->q and s->k in place
//        dump_matrix(s->q, dim, "SQROPE");
//        dump_matrix(s->k, kv_dim, "SKROPE");

        attn(p, s, head_size, pos, loff, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        matmul_l(s->xb2, s->xb, w->wo + ((uint32_t)l*dim*dim)*sizeof(float), dim, dim);

        // residual connection back into x
        for (uint8_t i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        // XXX64: xb is local, x is local, weight is remote
        rmsnorm(s->xb, x, w->rms_ffn_weight + ((uint32_t)l*dim)*sizeof(float), dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul_l(s->hb, s->xb, w->w1 + ((uint32_t)l*dim*hidden_dim)*sizeof(float), dim, hidden_dim);
        matmul_l(s->hb2, s->xb, w->w3 + ((uint32_t)l*dim*hidden_dim)*sizeof(float), dim, hidden_dim);

        // SwiGLU non-linearity
        for (uint8_t i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + exp(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul_l(s->xb, s->hb, w->w2 + ((uint32_t)l*dim*hidden_dim)*sizeof(float), hidden_dim, dim);

        // residual connection
        for (uint8_t i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    // XXX64: x is local, x is local, weight is remote
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul_ll(s->logits, x, w->wcls, p->dim, p->vocab_size);
//    dump_matrix_local(s->logits, p->vocab_size, "LOGITS(FORWARD)");
    return s->logits;
}
