MPICXX = mpic++
CXXFLAGS = -O3 -Wall

HEADERS = functions.h
BASE_SRC = spgemm.cpp distribute.cpp
BASE_OBJ = $(BASE_SRC:.cpp=.o)
BIN_SRC = pa3.cpp
BIN_OBJ = $(BIN_SRC:.cpp=.o)
BIN = $(BIN_SRC:.cpp=)

all: $(BIN)
.PHONY: all

$(BIN_OBJ) $(BASE_OBJ): $(HEADERS)
$(BIN_OBJ): $(BASE_OBJ)
$(BIN_OBJ) $(BASE_OBJ): %.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $<
$(BIN): %: %.o
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(BASE_OBJ)

clean:
	rm -f $(BIN) *.o
.PHONY: clean
