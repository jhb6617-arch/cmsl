SHELL := /bin/sh

NVCC := nvcc -ccbin $(CC)

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

TARGET := Pfmhzo.exe

# Compile flags
NVCCFLAGS := -O2 -std=c++14 -I$(INC_DIR) -dc \
             -Xcompiler -MMD -Xcompiler -MP \
             -Xcompiler -Wno-deprecated-declarations

# Link flags
LDFLAGS := -lcurand -lcufft -lcublas -lcusparse -lm -lstdc++

SOURCES := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

remove:
	rm -rf pol_data* *.vtk params* Charateristic* Read* *.out
