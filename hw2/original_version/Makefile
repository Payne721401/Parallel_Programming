# The two starters, untouched, built against the fixed files of each problem directory. They are
# the references for cases we write ourselves (spec 2.7): hw2-1-seq and hw2-2-seq are correct at
# small scale, just slow and memory-hungry. The tests under ../tests build them through here.
CXX      = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native

P1 = ../hw2-1
P2 = ../hw2-2

all: hw2-1-seq hw2-2-seq

hw2-1-seq: hw2-1.cpp $(P1)/main.cpp $(P1)/graph_gen.cpp $(P1)/graph_gen.hpp
	$(CXX) $(CXXFLAGS) -I$(P1) -o $@ $(P1)/main.cpp $(P1)/graph_gen.cpp hw2-1.cpp

hw2-2-seq: hw2-2.cpp $(P2)/main.cpp $(P2)/spgemm.hpp $(P2)/local_kernel.hpp
	$(CXX) $(CXXFLAGS) -I$(P2) -o $@ $(P2)/main.cpp hw2-2.cpp

clean:
	rm -f hw2-1-seq hw2-2-seq

.PHONY: all clean
