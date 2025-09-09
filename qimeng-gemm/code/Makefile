SRC := main.cpp gemm.cpp
TARGET := gemm

CXX := hipcc
CXXFLAGS := -O0 -std=c++11

all:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: all
	./$(TARGET)

clean:
	@$(RM) -v $(TARGET)
