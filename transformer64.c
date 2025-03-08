/* Inference for Llama-2 Transformer model in pure C */

#include "transformer64.h"

REUPtr reu_base = (REUPtr)0; // base address of weights.bin inside REU

const unsigned char config_bin[] = {
    #embed "config.bin"
};

// ----------------------------------------------------------------------------
// REU functions (access to transformer weights)

struct REU
{
    volatile uint8_t status;
    volatile uint8_t command;
    volatile uint16_t c64_base;
    volatile uint16_t reu_base;
    volatile uint8_t reu_base_bank;
    volatile uint16_t length;
    volatile uint8_t irq;
    volatile uint8_t control;
};

#define reu     (*((struct REU *)0xdf00))

void REU_init() {
    reu.control = 0; // increment both addresses
}

void REU_getf(REUPtr ptr, volatile float* out, uint16_t size) {
    reu.c64_base = (uint16_t)out;
    reu.reu_base = (uint16_t)(ptr & 0xFFFF);
    reu.reu_base_bank = (uint8_t)((ptr >> 16) & 0xFF);
    reu.length = size;
    reu.command = 0x91; // read from REU, execute immediately
}

void REU_putf(REUPtr ptr, volatile float* in, uint16_t size) {
    reu.c64_base = (uint16_t)in;
    reu.reu_base = (uint16_t)(ptr & 0xFFFF);
    reu.reu_base_bank = (uint8_t)((ptr >> 16) & 0xFF);
    reu.length = size;
    reu.command = 0x90; // write to REU, execute immediately
}

// ----------------------------------------------------------------------------
// Transformer model

void malloc_run_state(Transformer* t) {

    RunState64* s = &t->state;
    Config64* p = t->config;

    printf("Malloc\n");
    printf("p->dim=%d\n", p->dim);
    printf("p->n_heads=%d\n", p->n_heads);

    // we calloc instead of malloc to keep valgrind happy
    uint32_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    printf("Allocating x: %u bytes\n", p->dim * sizeof(float));
    s->x = calloc(p->dim, sizeof(float));
    printf("Allocating xb: %u bytes\n", p->dim * sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    printf("Allocating xb2: %u bytes\n", p->dim * sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    printf("Allocating hb: %u bytes\n", p->hidden_dim * sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    printf("Allocating hb2: %u bytes\n", p->hidden_dim * sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    printf("Allocating q: %u bytes\n", p->dim * sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    printf("Allocating key_cache: %lu bytes [TOOBIG]\n", p->n_layers * p->seq_len * kv_dim * sizeof(float));
//    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->key_cache = reu_base;
    reu_base += p->n_layers * p->seq_len * kv_dim * sizeof(float);
    printf("Allocating value_cache: %lu bytes [TOOBIG]\n", p->n_layers * p->seq_len * kv_dim * sizeof(float));
//    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = reu_base;
    reu_base += p->n_layers * p->seq_len * kv_dim * sizeof(float);
    printf("Allocating att: %u bytes [ALMOSTTOOBIG]\n", p->n_heads * p->seq_len * sizeof(float));
//    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->att = reu_base;
    reu_base += p->n_heads * p->seq_len * sizeof(float);
    printf("Allocating logits: %u bytes\n", p->vocab_size * sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    printf("Final reu_base=%lu\n", reu_base);
}

void memory_map_weights(Transformer* t) {
    TransformerWeights64* w = &t->weights;
    Config64* p = t->config;
    uint16_t shared_weights = p->shared_weights;
    REUPtr ptr = reu_base;

    printf("mmap start ptr=%lu\n", ptr);
    printf("p->dim=%d\n", p->dim);
    printf("p->n_heads=%d\n", p->n_heads);

    uint32_t head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 32 bit at least
    uint32_t n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += sizeof(float) * (uint32_t)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += sizeof(float) * n_layers * p->dim;
    w->wq = ptr;
    printf("wq(1)=%lu\n", w->wq);
    ptr += sizeof(float) * n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += sizeof(float) * n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += sizeof(float) * n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += sizeof(float) * n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += sizeof(float) * n_layers * p->dim;
    w->w1 = ptr;
    ptr += sizeof(float) * n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += sizeof(float) * n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += sizeof(float) * n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += sizeof(float) * p->dim;
    ptr += sizeof(float) * p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += sizeof(float) * p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
    printf("mmap end ptr=%lu\n", ptr);
    reu_base = ptr; // first free byte after weights (must match weights.bin length + initial offset)
}

void load_transformer(Transformer *t) {

    t->config = (Config64*) config_bin;
    REU_init();

//    REU_getf((uint32_t)0x20500, (float*)(0x0400), 1024);

    memory_map_weights(t);
    // allocate the RunState buffers
    malloc_run_state(t);

    volatile float f;
    f = 0.0;
    printf("wq(2)=%lu\n", t->weights.wq);
    REU_getf(t->weights.wq+sizeof(float)*0, &f, sizeof(float));
    printf("wq[0]=%f\n", f);
    REU_getf(t->weights.wq+sizeof(float)*3, &f, sizeof(float));
    printf("wq[3]=%f\n", f);
}