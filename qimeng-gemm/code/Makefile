SRC := main.cpp gemm.cpp
TARGET := gemm

CXX := hipcc
CXXFLAGS := -O0 -std=c++11

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test bench: $(TARGET)
	./$< $@

clean:
	@$(RM) -v $(TARGET)
