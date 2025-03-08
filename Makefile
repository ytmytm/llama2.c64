
TEMP:=0
NTOKENS:=256
SEED:=0

CFLAGS:=-g -Wall -c -O3
LDFLAGS:=-lm

test:	run runfull
	./runfull stories260K.bin -z tok512.bin -t $(TEMP) -s $(SEED) -n $(NTOKENS) -i "Once upon"
	./run stories260K.bin -z tok512.bin -t $(TEMP) -s $(SEED) -n $(NTOKENS) -i "Once upon"
	./run stories260K.bin -t $(TEMP) -s $(SEED) -n $(NTOKENS) -i "Once upon"

test-tokenizer: tokenizer.c tokenizer.h test-tokenizer.c util.c util.h
	gcc $(CFLAGS) tokenizer.c
	gcc $(CFLAGS) util.c
	gcc $(CFLAGS) test-tokenizer.c
	gcc test-tokenizer.o tokenizer.o util.o -o test-tokenizer $(LDFLAGS)
	./test-tokenizer

run:	run.c tokenizer.c tokenizer.h sampler.c sampler.h nnet.c nnet.h transformer.c transformer.h util.c util.h generate.c generate.h
	gcc $(CFLAGS) transformer.c
	gcc $(CFLAGS) tokenizer.c
	gcc $(CFLAGS) sampler.c
	gcc $(CFLAGS) nnet.c
	gcc $(CFLAGS) util.c
	gcc $(CFLAGS) generate.c
	gcc $(CFLAGS) run.c
	gcc run.o tokenizer.o sampler.o nnet.o transformer.o util.o generate.o -o run $(LDFLAGS)

runfull:	runfull.c
	gcc -g runfull.c -o runfull -lm

all: runfull run test

clean:
	rm -f run runfull *.o

