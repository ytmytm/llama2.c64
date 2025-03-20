/* Inference for Llama-2 Transformer model in pure C */

#include <math.h>
#include <string.h>

#include "nnet64.h"

// ----------------------------------------------------------------------------
// cache

float *wifbuf; // weight matrix buffer for matmul
float *xobuf;  // general output buffer for matmul

void nnet_init(Transformer* transformer) {
    Config64* p = transformer->config;
    uint8_t maxdim = p->hidden_dim;
    uint8_t dim = p->dim; // 64?

    // just in case
    if (p->dim > maxdim) { maxdim = p->dim; }
    if (((p->dim * p->n_kv_heads) / p->n_heads) > maxdim) { maxdim = (p->dim * p->n_kv_heads) / p->n_heads; }

    wifbuf = (float*)malloc(maxdim*sizeof(float));
    xobuf = (float*)malloc(dim*sizeof(float));
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, REUPtr weight, uint8_t size) {
    float *wif = xobuf;
    float *xi = x;
    float *oi = o;
    // calculate sum of squares
    float ss = 0.0;
    for (uint8_t j = 0; j < size; j++) {
        ss += (*xi)*(*xi);
        xi++;
    }
    ss /= size;
    ss += 0.00001;
    ss = 1.0 / sqrt(ss);
    // normalize and scale
    REU_getf(weight, xobuf, size*sizeof(float));
    xi = x;
    for (uint8_t j = 0; j < size; j++) {
        (*oi) = (*wif) * ss * (*xi);
        oi++;
        wif++;
        xi++;
    }
}

// x is remote, size is sampler->vocab_size (uint_16t)
void softmax(REUPtr x, uint16_t size) {
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
        xif = my_exp(xif - max_val);
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
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    float *xof = xobuf;
    float *wif;
    float *xi;
    for (uint8_t i = 0; i < d; i++) {
        (*xof) = 0.0;
        xi = x;
        REU_getf(w, wifbuf, n*sizeof(float));
        w += n*sizeof(float);
        wif = wifbuf;
        for (uint8_t j = 0; j < n; j++) {
            (*xof) += (*wif) * (*xi);
            wif++;
            xi++;
        }
        xof++;
    }
    REU_putf(xout, xobuf, d*sizeof(float));
}

// xout is local, x is local, w is remote, n/d are always dim/hidden_dim
void matmul_l(float* xout, float* x, REUPtr w, uint8_t n, uint8_t d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    float *xo = xout;
    float *wif;
    float *xi;
    for (uint8_t i = 0; i < d; i++) {
        (*xo) = 0.0;
        xi = x;
        REU_getf(w, wifbuf, n*sizeof(float));
        w += n*sizeof(float);
        wif = wifbuf;
        for (uint8_t j = 0; j < n; j++) {
            (*xo) += (*wif) * (*xi);
            wif++;
            xi++;
        }
        xo++;
    }
}

// xout is local, x is local, w is remote, n/d are always dim/vocab_size
void matmul_ll(float* xout, float* x, REUPtr w, uint8_t n, uint16_t d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    float *xo = xout;
    float *wif;
    float *xi;
    for (uint16_t i = 0; i < d; i++) {
        (*xo) = 0.0;
        xi = x;
        REU_getf(w, wifbuf, n*sizeof(float));
        w += n*sizeof(float);
        wif = wifbuf;
        for (uint8_t j = 0; j < n; j++) {
            (*xo) += (*wif) * (*xi);
            wif++;
            xi++;
        }
        xo++;
    }
}
    
void rope(uint8_t dim, RunState64 *s, uint8_t head_size, uint16_t pos, uint8_t kv_dim)
{
    static uint16_t last_pos = -1;
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    REUPtr vecq = s->q; // the vector to rotate (query or key)
    REUPtr veck = s->k; // the vector to rotate (query or key)
    float vi[2];
    float vo[2];
    float val = pos;
    float *fcir_table = s->fcir; // cache space

    if (last_pos != pos) {
        last_pos = pos;
        // cache the sin/cos values for the relative positional encoding
        for (uint8_t h = 0; h < head_size; h+=2) {
            fcir_table[h] = my_cos(val);
            fcir_table[h+1] = my_sin(val);
            val /= 10.0;
        }
    }

    uint8_t table_idx = 0;
    for (uint8_t i = 0; i < dim; i += 2)
    {
        float fcr = fcir_table[table_idx];
        float fci = fcir_table[table_idx + 1];
        uint8_t rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (uint8_t v = 0; v < rotn; v++)
        {
            REUPtr vec = v == 0 ? vecq : veck; // the vector to rotate (query or key)
            REU_getf(vec, &vi[0], 2 * sizeof(float));
            vo[0] = vi[0] * fcr - vi[1] * fci;
            vo[1] = vi[0] * fci + vi[1] * fcr;
            REU_putf(vec, &vo[0], 2 * sizeof(float));
        }
        vecq += 2 * sizeof(float);
        veck += 2 * sizeof(float);
        table_idx += 2;
        if (table_idx == head_size) { table_idx = 0; }
    }
}

void attn(Config64 *p, RunState64 *s, uint8_t head_size, uint16_t pos, uint32_t loff, uint8_t kv_dim, uint8_t kv_mul)
{
    // multihead attention. iterate over all heads
    for (uint8_t h = 0; h < p->n_heads; h++)
    {
        // get the query vector for this head
        REUPtr q = s->q + ((uint32_t)h * head_size) * sizeof(float); // XXX64: q is remote
        // attention scores for this head
        REUPtr att = s->att + ((uint32_t)h * p->seq_len) * sizeof(float); // XXX64: att is remote
        // iterate over all timesteps, including the current one
        REUPtr atti = att;
        for (uint16_t t = 0; t <= pos; t++)
        {
            // get the key vector for this head and at this timestep
            REUPtr qq = q;
            REUPtr k = s->key_cache + ((uint32_t)loff + (uint32_t)t * kv_dim + (uint32_t)(h / kv_mul) * head_size) * sizeof(float); // XXX64: key_cache is remote
            // calculate the attention score as the dot product of q and k
            float score = 0.0;
            float qi, ki;
            for (uint8_t i = 0; i < head_size; i++)
            {
                REU_getf(qq, &qi, sizeof(float)); // XXX can buffer whole thing
                REU_getf(k, &ki, sizeof(float));
                score += qi * ki;
                qq += sizeof(float);
                k += sizeof(float);
            }
            score /= sqrt(head_size);
            // save the score to the attention buffer
            REU_putf(atti, &score, sizeof(float)); // XXX buffer
            atti += sizeof(float);
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);

        // weighted sum of the values, store back into xb
        float *xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        atti = att;
        for (uint16_t t = 0; t <= pos; t++)
        {
            // get the value vector for this head and at this timestep
            REUPtr v = s->value_cache + ((uint32_t)loff + (uint32_t)t * kv_dim + (uint32_t)(h / kv_mul) * head_size) * sizeof(float);
            // get the attention weight for this timestep
            float a;
            REU_getf(atti, &a, sizeof(float));
            atti += sizeof(float);
            float vi;
            // accumulate the weighted value into xb
            for (uint8_t i = 0; i < head_size; i++)
            {
                REU_getf(v, &vi, sizeof(float));
                xb[i] += a * vi;
                v += sizeof(float);
            }
        }
    }
}

char ui_statusbuf[40];

// assumption: n_heads, dim, hidden_dim are <256
float* forward(Transformer* transformer, uint16_t token, uint16_t pos) {

    // a few convenience variables
    Config64* p = transformer->config;
    TransformerWeights64* w = &transformer->weights; // XXX64:all are remote
    RunState64* s = &transformer->state;
    float *x = s->x; // XXX64: x, s->x local
    uint8_t dim = p->dim;
    uint8_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    uint8_t kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    uint8_t hidden_dim =  p->hidden_dim;
    uint8_t head_size = dim / p->n_heads;

    // copy the token embedding into x
    // XXX64: token_embedding_table is remote, x is local
    REUPtr content_row = w->token_embedding_table + ((uint32_t)token * dim)*sizeof(float);
    REU_getf(content_row, x, dim*sizeof(float));

    // forward all the layers
    for(uint8_t l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        // XXX64: xb is local, x is local, weight is remote
        sprintf(ui_statusbuf, "layer %d rmsnorm1 [%d]", l, dim);
        ui_settopstatus(ui_statusbuf);
        rmsnorm(s->xb, x, w->rms_att_weight + ((uint32_t)l*dim)*sizeof(float), dim);

        // key and value point to the kv cache
        uint32_t loff = (uint32_t)l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + (loff + (uint32_t)pos * kv_dim)*sizeof(float);
        s->v = s->value_cache + (loff + (uint32_t)pos * kv_dim)*sizeof(float);

        // qkv matmuls for this position
        sprintf(ui_statusbuf, "layer %d matrix1 [%d*%d]", l, dim, dim);
        ui_settopstatus(ui_statusbuf);
        matmul(s->q, s->xb, w->wq + ((uint32_t)l*dim*dim)*sizeof(float), dim, dim);
        sprintf(ui_statusbuf, "layer %d matrix2 [%d*%d]", l, dim, kv_dim);
        ui_settopstatus(ui_statusbuf);
        matmul(s->k, s->xb, w->wk + ((uint32_t)l*dim*kv_dim)*sizeof(float), dim, kv_dim);
        sprintf(ui_statusbuf, "layer %d matrix3 [%d*%d]", l, dim, kv_dim);
        ui_settopstatus(ui_statusbuf);
        matmul(s->v, s->xb, w->wv + ((uint32_t)l*dim*kv_dim)*sizeof(float), dim, kv_dim);

        sprintf(ui_statusbuf, "layer %d rope [%d]", l, dim);
        ui_settopstatus(ui_statusbuf);
        rope(dim, s, head_size, pos, kv_dim); // modifies s->q and s->k in place

        sprintf(ui_statusbuf, "layer %d attention [%d]", l, kv_dim);
        ui_settopstatus(ui_statusbuf);
        attn(p, s, head_size, pos, loff, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        sprintf(ui_statusbuf, "layer %d matrix4 [%d*%d]", l, dim, dim);
        ui_settopstatus(ui_statusbuf);
        matmul_l(s->xb2, s->xb, w->wo + ((uint32_t)l*dim*dim)*sizeof(float), dim, dim);

        // residual connection back into x
        for (uint8_t i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        // XXX64: xb is local, x is local, weight is remote
        sprintf(ui_statusbuf, "layer %d rmsnorm2 [%d]", l, dim);
        ui_settopstatus(ui_statusbuf);
        rmsnorm(s->xb, x, w->rms_ffn_weight + ((uint32_t)l*dim)*sizeof(float), dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        sprintf(ui_statusbuf, "layer %d matrix6 [%d*%d]", l, dim, hidden_dim);
        ui_settopstatus(ui_statusbuf);
        matmul_l(s->hb, s->xb, w->w1 + ((uint32_t)l*dim*hidden_dim)*sizeof(float), dim, hidden_dim);
        sprintf(ui_statusbuf, "layer %d matrix7 [%d*%d]", l, dim, hidden_dim);
        ui_settopstatus(ui_statusbuf);
        matmul_l(s->hb2, s->xb, w->w3 + ((uint32_t)l*dim*hidden_dim)*sizeof(float), dim, hidden_dim);

        // SwiGLU non-linearity
        sprintf(ui_statusbuf, "layer %d swiglu [%d]", l, hidden_dim);
        ui_settopstatus(ui_statusbuf);
        for (uint8_t i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + my_exp(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        sprintf(ui_statusbuf, "layer %d matrix8 [%d*%d]", l, hidden_dim, dim);
        ui_settopstatus(ui_statusbuf);
        matmul_l(s->xb, s->hb, w->w2 + ((uint32_t)l*dim*hidden_dim)*sizeof(float), hidden_dim, dim);

        // residual connection
        for (uint8_t i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    // XXX64: x is local, x is local, weight is remote
    sprintf(ui_statusbuf, "layer - rmsnorm3 [%d]", dim);
    ui_settopstatus(ui_statusbuf);
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    sprintf(ui_statusbuf, "layer - matrix9 [%d*%d]", dim, p->vocab_size);
    ui_settopstatus(ui_statusbuf);
    matmul_ll(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}
