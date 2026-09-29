# Builds hw1-1.cpp, hw1-2.cpp and hw1-3.cpp, each into its own program. Submit this file
# to change the compilation flags, or to add sources and libraries of your own.
CXX = g++

# -march decides which instructions GCC may emit, so getting it wrong is a
# crash, not a slowdown: a binary built for a newer CPU than it runs on dies
# with SIGILL. native means "target exactly the machine I am compiling on",
# which is only safe once the login node and the compute nodes are known to
# match. Verified 2026-09-29 -- both report Intel Xeon Platinum 8352V:
#   lscpu | grep 'Model name'
#   srun -n 1 -c 8 lscpu | grep 'Model name'
# If they ever diverge, fall back to -march=x86-64-v3, the psABI level that
# guarantees AVX2 + FMA on any CPU from 2013 onwards:
#   make ARCH=-march=x86-64-v3
# Measured effect: hw1-2's blur loops go from 16-byte vectors (SSE2, 2 doubles)
# to 32-byte (AVX2, 4 doubles).
ARCH ?= -march=native -mtune=native

# -fno-math-errno lets sqrt and exp compile to instructions rather than library
# calls that have to set errno on a domain error. It changes no computed value.
# Deliberately NOT -ffast-math: that implies -fassociative-math, which lets the
# compiler regroup floating-point sums. Reassociation would put hw1-2's
# +-0.02 rad tolerance and hw1-3's 1e-9 relative tolerance at risk, and hw1-3
# compounds its error over as many as 1000 time steps.
CXXFLAGS = -std=c++11 -O3 $(ARCH) -fno-math-errno -pthread -fopenmp
LDLIBS = -lpng

all: hw1-1 hw1-2 hw1-3

# hw1-3 does no image I/O.
hw1-3: LDLIBS =

hw1-1 hw1-2 hw1-3: %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f hw1-1 hw1-2 hw1-3

.PHONY: all clean
