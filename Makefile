BLAS = ../BLAS-3.12.0/blas_LINUX.a
LAPACK = ../lapack-3.12.0/liblapack.a
EMCC = emcc
EMFC = ../build/bin/flang-new
FORTRAN_RUNTIME = ../build/flang/runtime/libFortranRuntime.a
OBJECTS = classifier.o

CFLAGS = -O2
LDFLAGS = -s ALLOW_MEMORY_GROWTH=1 -s STACK_SIZE=4MB
LDFLAGS += -sEXPORTED_FUNCTIONS=_classifier_,_malloc,_free
LDFLAGS += $(BLAS) $(LAPACK) $(FORTRAN_RUNTIME)

all: www/mnist.data www/mnist.js

www/mnist.js: $(OBJECTS)
	$(EMCC) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@

www/mnist.data: train/train.py
	python train/train.py
	mv mnist.data www/mnist.data

%.o: %.f90
	$(EMFC) -o $@ -c $<

clean:
	rm -f $(OBJECTS) www/mnist.js
