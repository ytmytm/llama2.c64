#!/bin/sh

/home/maciej/Projekty/espllm/oscar64/bin/oscar64 -O2 llama2c64.c \
 && x64 -warp -reu -reusize 2048 -reuimage weights.bin llama2c64.prg

# ./run stories260K.bin -z tok512.bin -i "Once upon a time" -t 0 -n 16
