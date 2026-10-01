# Builds hw1-1.cpp, hw1-2.cpp and hw1-3.cpp, each into its own program.
CXX = g++

# -march decides which instructions GCC may emit, so getting it wrong is a
# SIGILL rather than a slowdown. native is only safe while the login node and
# the compute nodes match -- verified 2026-09-29, both Intel Xeon Platinum
# 8352V, via `lscpu | grep 'Model name'` with and without srun. If they ever
# diverge: make ARCH=-march=x86-64-v3 (the psABI level that guarantees AVX2 and
# FMA). Measured: hw1-2's blur loops go from 16-byte SSE2 to 32-byte AVX2.
ARCH ?= -march=native -mtune=native

# -fno-math-errno: sqrt and exp become instructions rather than library calls
# that must set errno on a domain error. It changes no computed value.
#
# -ffp-contract=off is here for correctness, not speed. At GCC's C++ default of
# "fast" the compiler fuses a*b + c into one FMA that rounds once instead of
# twice -- more accurate, but not the arithmetic the reference performed, and
# both hw1-2 and hw1-3 are judged against its exact output. Measured: hw1-2
# gained 8 matches on b08 and 3 on a08, and the checker demands the count agree
# exactly; hw1-3's p01 and p03 went WA, because react() takes a fixed three
# Newton iterations, so a perturbed intermediate is never corrected and instead
# compounds over every step. p02, whose seed yields no inclusions and so never
# calls react(), stayed AC. Vector width is untouched -- the wide-vector
# register count is identical either way.
#
# Not -ffast-math either: -fassociative-math would let the compiler regroup the
# sums as well, breaking the same two tolerances by a much wider margin.
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
