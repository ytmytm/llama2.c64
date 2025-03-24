#!/usr/bin/env python3

import mmap
import struct

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
    tokenizer = Tokenizer()
    tokenizer.build_tokenizer("tok512.bin", 512)
    tokenizer.save_tokenizer("tokenizer-py.bin")
    