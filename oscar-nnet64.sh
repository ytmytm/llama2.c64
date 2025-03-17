#!/bin/sh

/home/maciej/Projekty/espllm/oscar64/bin/oscar64 test-nnet64.c \
 && x64 -warp -reu -reusize 2048 -reuimage weights.bin test-nnet64.prg
