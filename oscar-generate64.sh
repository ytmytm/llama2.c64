#!/bin/sh

/home/maciej/Projekty/espllm/oscar64/bin/oscar64 test-generate64.c \
 && x64 -reu -reusize 2048 -reuimage weights.bin test-generate64.prg

# ./run stories260K.bin -z tok512.bin -i "Once upon a time" -t 0 -n 16
