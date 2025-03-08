/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "transformer.h"

// ----------------------------------------------------------------------------
// Transformer model

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    printf("p->dim=%d\n", p->dim);
    printf("p->n_heads=%d\n", p->n_heads);
    printf("p->n_kv_heads=%d\n", p->n_kv_heads);
    printf("kv_dim=%d\n", kv_dim);
    printf("p->hidden_dim=%d\n", p->hidden_dim);
    printf("p->n_layers=%d\n",p->n_layers);    
    printf("p->seq_len=%d\n", p->seq_len);
    printf("Allocating x: %zu bytes\n", p->dim * sizeof(float));
    s->x = calloc(p->dim, sizeof(float));
    printf("Allocating xb: %zu bytes\n", p->dim * sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    printf("Allocating xb2: %zu bytes\n", p->dim * sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    printf("Allocating hb: %zu bytes\n", p->hidden_dim * sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    printf("Allocating hb2: %zu bytes\n", p->hidden_dim * sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    printf("Allocating q: %zu bytes\n", p->dim * sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    printf("Allocating key_cache: %zu bytes [TOOBIG]\n", p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    printf("Allocating value_cache: %zu bytes [TOOBIG]\n", p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    printf("Allocating att: %zu bytes [ALMOSTTOOBIG]\n", p->n_heads * p->seq_len * sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    printf("Allocating logits: %zu bytes\n", p->vocab_size * sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    printf("dim: %d\n", config->dim);

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
    //
    // write out config in 16-bit ints
    file = fopen("config.bin", "wb");
    int16_t i16;
    i16 = config->dim;
    fwrite(&i16, sizeof(int16_t), 1, file);
    i16 = config->hidden_dim;
    fwrite(&i16, sizeof(int16_t), 1, file);
    i16 = config->n_layers;
    fwrite(&i16, sizeof(int16_t), 1, file);
    i16 = config->n_heads;
    fwrite(&i16, sizeof(int16_t), 1, file);
    i16 = config->n_kv_heads;
    fwrite(&i16, sizeof(int16_t), 1, file);
    // could skip vocab_size, it's in tokenizer.bin
    i16 = config->vocab_size;
    fwrite(&i16, sizeof(int16_t), 1, file);
    i16 = config->seq_len;
    fwrite(&i16, sizeof(int16_t), 1, file);
    i16 = shared_weights;
    fwrite(&i16, sizeof(int16_t), 1, file);
    fclose(file);
    // info about weights
    printf("skip %lu bytes from weights.bin\n",sizeof(Config));
    printf("weights_ptr=%x %x %x\n",((unsigned char*)weights_ptr)[0],((unsigned char*)weights_ptr)[1],((unsigned char*)weights_ptr)[2]);
    printf("shared_weights=%d\n",shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);

    printf("wq[0]=%f\n", t->weights.wq[0]);
    printf("wq[3]=%f\n", t->weights.wq[3]);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}
