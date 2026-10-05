# Builds lab2.cpp. `module load openmpi` first -- mpicxx is the module's.
# Submit this file to change the compilation flags, or to add sources and
# libraries of your own.
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3

all: lab2

lab2: lab2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f lab2

.PHONY: all clean
