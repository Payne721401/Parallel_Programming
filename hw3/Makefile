# HW 3-1. The build the judge does, for you on Nano5: it compiles hw3-1.cu (yours) with the fixed
# main.cu, with exactly these flags. The judge does not read this Makefile; it compiles your
# hw3-1.cu itself, against its own copy of main.cu and ca.hpp.
#
# On Nano5 it names CUDA 12.4 and gcc 12.5 by path, as the judge does, so it needs no module
# loaded and builds the same on the login node and on a GPU node; elsewhere it takes the nvcc
# on PATH and its default host compiler.
CUDA_NVCC = /work/HPC_software/LMOD/nvidia/packages/cuda-12.4/bin/nvcc
GXX       = /work/HPC_software/LMOD/gcc/12.5.0/bin/g++
NVCC     ?= $(if $(wildcard $(CUDA_NVCC)),$(CUDA_NVCC),nvcc)
NVCCFLAGS = -std=c++17 -O3 -arch=sm_90
ifneq ($(wildcard $(GXX)),)
NVCCFLAGS += -ccbin $(GXX) -Xcompiler -static-libstdc++,-static-libgcc
endif

hw3-1: main.cu hw3-1.cu ca.hpp
	$(NVCC) $(NVCCFLAGS) main.cu hw3-1.cu -o $@

clean:
	rm -f hw3-1

.PHONY: clean
