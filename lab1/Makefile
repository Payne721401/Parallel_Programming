# Builds lab1.cpp. `module load openmpi` first -- mpicxx is the module's.
# Submit this file to change the compilation flags, or to add sources and
# libraries of your own.
CXX = mpicxx
CXXFLAGS = -std=c++11 -O3

all: lab1

lab1: lab1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f lab1

.PHONY: all clean
