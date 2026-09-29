# Builds hw1-1.cpp, hw1-2.cpp and hw1-3.cpp, each into its own program. Submit this file
# to change the compilation flags, or to add sources and libraries of your own.
CXX = g++
CXXFLAGS = -std=c++11 -O3 -pthread -fopenmp
LDLIBS = -lpng

all: hw1-1 hw1-2 hw1-3

hw1-1 hw1-2 hw1-3: %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f hw1-1 hw1-2 hw1-3

.PHONY: all clean
