CC = oscar64
VICE = x64
REU_SIZE = 2048
REU_IMAGE = weights.reu
PROGRAM = llama2c64.prg
PROGRAM_EXO = llama2exo.prg
SOURCE = llama2c64.c
HEADERS = tokenizer64.h transformer64.h nnet64.h sampler64.h util.h generate64.h
SOURCES = ui64.c math.c tokenizer64.c transformer64.c nnet64.c sampler64.c util64.c generate64.c
MODEL_FILES = $(REU_IMAGE) config.bin tokenizer.bin
INPUT_MODEL = stories260K.bin
INPUT_TOKENIZER = tok512.bin
EXOMIZER = exomizer

.PHONY: all build test clean love

all: build

build: $(PROGRAM)

$(PROGRAM): $(SOURCE) $(HEADERS) $(SOURCES) $(MODEL_FILES)
	$(CC) -O2 $(SOURCE)
	$(EXOMIZER) sfx basic $(PROGRAM) -o $(PROGRAM_EXO)

$(MODEL_FILES): generate-model-files.py $(INPUT_MODEL) $(INPUT_TOKENIZER)
	python3 generate-model-files.py --checkpoint $(INPUT_MODEL) --tokenizer $(INPUT_TOKENIZER)

test: $(PROGRAM)
	$(VICE) -warp -reu -reusize $(REU_SIZE) -reuimage $(REU_IMAGE) $(PROGRAM)

love:
	@echo "Not war, eh?"

clean:
	rm -f $(PROGRAM) $(MODEL_FILES)
