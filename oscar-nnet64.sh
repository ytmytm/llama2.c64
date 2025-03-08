#!/bin/sh

/home/maciej/Projekty/espllm/oscar64/bin/oscar64 -e test-nnet64.c \
 && x64 -reu -reusize 2048 -reuimage weights.bin test-nnet64.prg
