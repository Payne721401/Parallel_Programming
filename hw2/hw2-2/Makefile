# HW 2-2. This is the build the judge does, for your own machine and the login node: it compiles
# hw2-2.cpp (yours) with the fixed files, with exactly these flags. The judge does not read
# this Makefile; it compiles your hw2-2.cpp itself, against its own copies of the fixed files.
CXX      = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native

all: hw2-2

hw2-2: main.cpp hw2-2.cpp spgemm.hpp local_kernel.hpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp hw2-2.cpp

clean:
	rm -f hw2-2

.PHONY: all clean
