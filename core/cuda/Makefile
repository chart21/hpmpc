# User-specified options
arch ?= sm_89
CUDA_PATH ?= /home/usr/local/cuda
CUTLASS_PATH ?= /home/ubuntu/cutlass
#replace with e.g. make arch=sm_52 CUDA_PATH=/opt/cuda CUTLASS_PATH=/home/christopher/workspace/cutlass

# Derived variables
NVCC=$(CUDA_PATH)/bin/nvcc
NVCCOPTs= -arch=$(arch) --expt-relaxed-constexpr -std=c++20
CUDALIB=$(CUDA_PATH)/lib64

BIN_DIR=bin
TARGETS= $(BIN_DIR)/gemm_cutlass_int.o $(BIN_DIR)/conv_cutlass_int_NCHW.o $(BIN_DIR)/conv_cutlass_int_CHWN.o

CUTLASSINC=$(CUTLASS_PATH)/include
CUTTOOLINC=$(CUTLASS_PATH)/tools/util/include
CUTLASSLIB=$(CUTLASS_PATH)/build/tools/library/

OPTION= -Iinclude/ -I$(CUTLASSINC) -I$(CUTTOOLINC)

JUNK=$(TARGETS) a.out

all: $(TARGETS)

$(BIN_DIR)/gemm_cutlass_int.o: gemm_cutlass_int.cu | $(BIN_DIR)
	$(NVCC) $< $(NVCCOPTs) $(OPTION) -o $@ -c -Xptxas -O3

$(BIN_DIR)/conv_cutlass_int_NCHW.o: conv_cutlass_int.cu | $(BIN_DIR)
	$(NVCC) $< $(NVCCOPTs) $(OPTION) -o $@ -c -Xptxas -O3 -DPIGEON_LAYOUT=\"NCHW\"

$(BIN_DIR)/conv_cutlass_int_CHWN.o: conv_cutlass_int.cu | $(BIN_DIR)
	$(NVCC) $< $(NVCCOPTs) $(OPTION) -o $@ -c -Xptxas -O3 -DPIGEON_LAYOUT=\"CHWN\"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(JUNK) $(BIN_DIR)

