
LIB_FLAGS = -L../ -larmadillo
#LIB_FLAGS = -lblas -llapack
#LIB_FLAGS = -lopenblas -llapack

#CXX_FLAGS = -I../tmp/include/ -std=c++14 -Wshadow -Wall -pedantic -O0
CXX_FLAGS = -I../tmp/include/ -std=c++14 -Wshadow -Wall -pedantic -Og
#CXX_FLAGS = -I../tmp/include/ -std=c++14 -Wshadow -Wall -pedantic -Og -fopenmp
#CXX_FLAGS = -I../tmp/include/ -std=c++14 -Wshadow -Wall -pedantic -Og -DARMA_DONT_USE_WRAPPER
#CXX_FLAGS = -I../tmp/include/ -std=c++14 -Wshadow -Wall -pedantic -Og -fsanitize=address -fsanitize=leak -fsanitize=undefined -fsanitize=bounds -fsanitize=bounds-strict -g
#CXX_FLAGS = -I../tmp/include/ -std=c++23 -Wshadow -Wall -pedantic -Og -march=native -DARMA_FORCE_USE_FP16


OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

main: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJECTS) $(LIB_FLAGS)


all: main

.PHONY: clean

clean:
	rm -f main *.o
