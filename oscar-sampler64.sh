#!/bin/sh

/home/maciej/Projekty/espllm/oscar64/bin/oscar64 -e test-sampler64.c \
 && x64 -reu -reusize 2048 -reuimage weights.bin test-sampler64.prg

