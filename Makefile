MPICXX=mpic++

CFLAGS=-std=c++11 -O2 -Wall -Werror
LDFLAGS=-lm
SRC_DIR=./src

all: clean build

build: MatrixConvolution

Matrix.o:$(SRC_DIR)/Matrix.cpp
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ -c $<

MatrixConvolution.o:$(SRC_DIR)/MatrixConvolution.cpp
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ -c $<

MatrixConvolution: MatrixConvolution.o  Matrix.o
	$(MPICXX) $(CFLAGS) $(LDFLAGS) -o $@ $+

clean:
	rm -rf *.o MatrixConvolution

run: mpirun -np 8 ./MatrixConvolution
