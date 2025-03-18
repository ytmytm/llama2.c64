
TEMP:=0
NTOKENS:=256
SEED:=0

CFLAGS:=-g -Wall -c -O3
LDFLAGS:=-lm

test:	run runfull
	./runfull stories260K.bin -z tok512.bin -t $(TEMP) -s $(SEED) -n $(NTOKENS) -i "Once upon"
	./run stories260K.bin -z tok512.bin -t $(TEMP) -s $(SEED) -n $(NTOKENS) -i "Once upon"
	./run stories260K.bin -t $(TEMP) -s $(SEED) -n $(NTOKENS) -i "Once upon"

test-sampler: sampler64.c sampler64.h test-sampler.c
	gcc $(CFLAGS) sampler64.c
	gcc $(CFLAGS) test-sampler.c
	gcc test-sampler.o sampler64.o -o test-sampler $(LDFLAGS)
	./test-sampler

test-tokenizer: tokenizer.c tokenizer.h test-tokenizer.c util.c util.h
	gcc $(CFLAGS) tokenizer.c
	gcc $(CFLAGS) util.c
	gcc $(CFLAGS) test-tokenizer.c
	gcc test-tokenizer.o tokenizer.o util.o -o test-tokenizer $(LDFLAGS)
	./test-tokenizer

test-nnet: transformer.c transformer.h nnet.c nnet.h util.c util.h test-nnet.c
	gcc $(CFLAGS) test-nnet.c
	gcc $(CFLAGS) transformer.c
	gcc $(CFLAGS) nnet.c
	gcc $(CFLAGS) util.c
	gcc test-nnet.o transformer.o nnet.o util.o -o test-nnet $(LDFLAGS)
	./test-nnet

test-generate:	test-generate.c tokenizer.c tokenizer.h sampler.c sampler.h nnet.c nnet.h transformer.c transformer.h util.c util.h generate.c generate.h
	gcc $(CFLAGS) transformer.c
	gcc $(CFLAGS) tokenizer.c
	gcc $(CFLAGS) sampler.c
	gcc $(CFLAGS) nnet.c
	gcc $(CFLAGS) util.c
	gcc $(CFLAGS) generate.c
	gcc $(CFLAGS) test-generate.c
	gcc test-generate.o tokenizer.o sampler.o nnet.o transformer.o util.o generate.o -o test-generate $(LDFLAGS)
	./test-generate stories260K.bin -z tok512.bin -i "Zoo" -t 0 -n 10

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

#all: runfull run test
all: test-nnet

clean:
	rm -f run runfull *.o

