BUILD_DIR ?= build

KERNEL ?= gemm.cpp
SRC := main.cpp $(KERNEL)
EXE := $(BUILD_DIR)/$(notdir $(basename $(KERNEL))).out

CXX := hipcc
CPPFLAGS := -MMD -MP -I.
CXXFLAGS := -O0 -std=c++11

all: $(EXE)

$(EXE): $(SRC) | $(dir $(EXE))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

-include $(EXE:.out=.d)

$(dir $(EXE)):
	mkdir -p $@

test bench: $(EXE)
	$< $@ $(ITERS) $(WARMUP)

clean:
	@$(RM) -rv $(BUILD_DIR)
