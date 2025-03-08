/* Inference for Llama-2 Transformer model in pure C */

#include <stdint.h>

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

// ----------------------------------------------------------------------------
// Transformer model

typedef uint32_t REUPtr;

typedef struct {
    uint16_t dim; // transformer dimension
    uint16_t hidden_dim; // for ffn layers
    uint16_t n_layers; // number of layers
    uint16_t n_heads; // number of query heads
    uint16_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    uint16_t vocab_size; // vocabulary size, usually 256 (byte-level)
    uint16_t seq_len; // max sequence length
    uint16_t shared_weights;
} Config64;

// this is all within REU, these are all float*
typedef struct {
    // token embedding table
    REUPtr token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    REUPtr rms_att_weight; // (layer, dim) rmsnorm weights
    REUPtr rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    REUPtr wq; // (layer, dim, n_heads * head_size)
    REUPtr wk; // (layer, dim, n_kv_heads * head_size)
    REUPtr wv; // (layer, dim, n_kv_heads * head_size)
    REUPtr wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    REUPtr w1; // (layer, hidden_dim, dim)
    REUPtr w2; // (layer, dim, hidden_dim)
    REUPtr w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    REUPtr rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    REUPtr wcls;
} TransformerWeights64;

// big arrays from here are in REU
typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
//    float *att; // buffer for scores/attention values (n_heads, seq_len)
    REUPtr att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
//    float* key_cache;   // (layer, seq_len, dim)
//    float* value_cache; // (layer, seq_len, dim)
    REUPtr key_cache;   // (layer, seq_len, dim)
    REUPtr value_cache; // (layer, seq_len, dim)
} RunState64;

typedef struct {
    Config64* config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights64 weights; // the weights of the model
    RunState64 state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
//    int fd; // file descriptor for memory mapping
//    float* data; // memory mapped data pointer
//    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void build_transformer(Transformer *t, char* checkpoint_path);
void free_transformer(Transformer* t);

#endif // TRANSFORMER_H