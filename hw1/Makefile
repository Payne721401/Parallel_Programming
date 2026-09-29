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
#
# -ffp-contract=off is required for correctness here. Left at GCC's C++ default
# of "fast", the compiler fuses a*b + c into one FMA instruction that rounds
# once instead of twice. That is more accurate, but it is not the arithmetic the
# reference implementations performed, and both hw1-2 and hw1-3 are judged
# against their exact output:
#   hw1-2 accepts a match when best < 0.75 * second, a discrete decision on a
#   continuous quantity, so a last-bit change flips whichever candidates sit on
#   that boundary. Measured: 8 extra matches on b08, 3 on a08, and hw1-2-check
#   requires the count to agree exactly.
#   hw1-3's react() runs a fixed three Newton iterations rather than iterating
#   to convergence, so a perturbed intermediate is never corrected -- it reaches
#   the output and then compounds over every time step. p01 (100 steps) and p03
#   (50 steps) both went WA; p02, whose seed yields zero inclusions and so never
#   calls react(), stayed AC.
# Vectorisation is untouched: the wide-vector register count is identical with
# and without this flag, so it costs the fusion, not the vector width.
#
# Deliberately NOT -ffast-math: that implies -fassociative-math, which lets the
# compiler regroup floating-point sums as well. Reassociation would break the
# same two tolerances by a much larger margin.
CXXFLAGS = -std=c++11 -O3 $(ARCH) -fno-math-errno -ffp-contract=off -pthread -fopenmp
LDLIBS = -lpng

all: hw1-1 hw1-2 hw1-3

# hw1-3 does no image I/O.
hw1-3: LDLIBS =

hw1-1 hw1-2 hw1-3: %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f hw1-1 hw1-2 hw1-3

.PHONY: all clean
