#!/bin/sh

/home/maciej/Projekty/espllm/oscar64/bin/oscar64 -e test-tokenizer64.c \
 && x64 -reu -reusize 2048 -reuimage weights.bin test-transformer64.prg

