#!/usr/bin/env python3

import mmap
import struct
import os
import argparse

class Weights:
    def __init__(self):
        self.weights_data = None

    def read_weights(self, checkpoint, output_filename="weights.bin"):
        with open(checkpoint, "rb") as file:
            file.seek(28)  # Skip the first 28 bytes (Config)
            self.weights_data = file.read()

        with open(output_filename, "wb") as file:
            file.write(self.weights_data)

        self.pad_to_next_multiple(output_filename)

    def pad_to_next_multiple(self, filename, multiples=(2, 4, 8, 16)):
        file_size = os.path.getsize(filename)
        next_multiple = min(m for m in multiples if m * 1024 * 1024 > file_size)
        padding_size = next_multiple * 1024 * 1024 - file_size

        with open(filename, "ab") as file:
            file.write(b'\0' * padding_size)
class Config:
    def __init__(self):
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0

    def read_checkpoint(self, checkpoint, output_filename="config.bin"):
        with open(checkpoint, "rb") as file:
            config_data = file.read(struct.calcsize('iiiiiii'))
            (self.dim, self.hidden_dim, self.n_layers, self.n_heads, 
             self.n_kv_heads, self.vocab_size, self.seq_len) = struct.unpack('iiiiiii', config_data)

            shared_weights = self.vocab_size > 0
            self.vocab_size = abs(self.vocab_size)

        with open(output_filename, "wb") as file:
            file.write(struct.pack('h', self.dim))
            file.write(struct.pack('h', self.hidden_dim))
            file.write(struct.pack('h', self.n_layers))
            file.write(struct.pack('h', self.n_heads))
            file.write(struct.pack('h', self.n_kv_heads))
            file.write(struct.pack('h', self.vocab_size))
            file.write(struct.pack('h', self.seq_len))
            file.write(struct.pack('h', int(shared_weights)))

class Tokenizer:
    def __init__(self):
        self.vocab = []
        self.vocab_scores = []
        self.sorted_vocab = []
        self.vocab_size = 0
        self.max_token_length = 0
        self.byte_pieces = [bytes([i]) for i in range(256)]
        self.str_buffer = None

    def build_tokenizer(self, tokenizer_path, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = [None] * vocab_size
        self.vocab_scores = [0.0] * vocab_size
        self.sorted_vocab = [None] * vocab_size

        with open(tokenizer_path, "rb") as file:
            self.max_token_length = struct.unpack('i', file.read(4))[0]
            for i in range(vocab_size):
                self.vocab_scores[i] = struct.unpack('f', file.read(4))[0]
                len_str = struct.unpack('i', file.read(4))[0]
                self.vocab[i] = bytearray(file.read(len_str))

        self.sorted_vocab = sorted([(self.vocab[i], i) for i in range(vocab_size)], key=lambda x: x[0])
        self.str_buffer = bytearray((self.max_token_length * 2 + 1 + 2))

    def save_tokenizer(self, save_path):
        with open(save_path, "wb") as file:
            # Write vocab_size as uint16_t
            file.write(struct.pack('H', self.vocab_size))
            
            # Write vocab_scores as float
            for score in self.vocab_scores:
                file.write(struct.pack('f', score))
            
            # Write lengths of vocab strings as uint8_t
            for token in self.vocab:
                file.write(struct.pack('B', len(token) + 1))
            
            # Write sorted_vocab ids as uint16_t
            for _, id in self.sorted_vocab:
                file.write(struct.pack('H', id))
            
            # Write vocab strings with null terminator
            for token in self.vocab:
                file.write(token + b'\0')

    def load_tokenizer(self, load_path):
        with open(load_path, "rb") as f:
            self.mmap_size = f.seek(0, 2)
            f.seek(0)
            self.mmap_ptr = mmap.mmap(f.fileno(), self.mmap_size, access=mmap.ACCESS_READ)

        ptr = 0
        self.max_token_length = struct.unpack_from('i', self.mmap_ptr, ptr)[0]
        ptr += 4
        self.vocab_size = struct.unpack_from('i', self.mmap_ptr, ptr)[0]
        ptr += 4

        self.vocab = [None] * self.vocab_size
        self.vocab_scores = [0.0] * self.vocab_size
        self.sorted_vocab = [None] * self.vocab_size

        for i in range(self.vocab_size):
            self.vocab_scores[i] = struct.unpack_from('f', self.mmap_ptr, ptr)[0]
            ptr += 4
            len_str = struct.unpack_from('i', self.mmap_ptr, ptr)[0]
            ptr += 4
            self.vocab[i] = bytearray(self.mmap_ptr[ptr:ptr + len_str - 1])
            ptr += len_str

        for i in range(self.vocab_size):
            id = struct.unpack_from('i', self.mmap_ptr, ptr)[0]
            ptr += 4
            self.sorted_vocab[i] = (self.vocab[id], id)

        self.str_buffer = bytearray((self.max_token_length * 2 + 1 + 2))

    def free_tokenizer(self):
        if hasattr(self, 'mmap_ptr'):
            self.mmap_ptr.close()
        else:
            for token in self.vocab:
                del token
        del self.vocab
        del self.vocab_scores
        del self.sorted_vocab
        del self.str_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model files from checkpoints and tokenizer data.")
    parser.add_argument("--checkpoint", default="stories260K.bin", help="Path to the model checkpoint file. Default is 'stories260K.bin'.")
    parser.add_argument("--tokenizer", default="tok512.bin", help="Path to the tokenizer file. Default is 'tok512.bin'.")
    args = parser.parse_args()

    config = Config()
    config.read_checkpoint(args.checkpoint, "config.bin")
    
    tokenizer = Tokenizer()
    tokenizer.build_tokenizer(args.tokenizer, config.vocab_size)
    tokenizer.save_tokenizer("tokenizer.bin")
    tokenizer.free_tokenizer()
    
    weights = Weights()
    weights.read_weights(args.checkpoint, "weights.bin")

    print(f"Tokenizer saved to tokenizer.bin")
    print(f"Config saved to config.bin")
    print(f"Weights saved as REU image to weights.bin")