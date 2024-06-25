# User-specified options
arch ?= sm_89
CUDA_PATH ?= /home/usr/local/cuda
CUTLASS_PATH ?= /home/ubuntu/cutlass
#replace with e.g. make arch=sm_52 CUDA_PATH=/opt/cuda CUDA_PATH=/home/christopher/workspace/cutlass

# Derived variables
NVCC=$(CUDA_PATH)/bin/nvcc
NVCCOPTs= -arch=$(arch) --expt-relaxed-constexpr -std=c++20
CUDALIB=$(CUDA_PATH)/lib64

TARGETS= gemm_cutlass_int.o conv_cutlass_int_NCHW.o conv_cutlass_int_CHWN.o

CUTLASSINC=$(CUDA_PATH)/include
CUTTOOLINC=$(CUDA_PATH)/tools/util/include
CUTLASSLIB=$(CUDA_PATH)/build/tools/library/

OPTION= -Iinclude/ -I$(CUTLASSINC) -I$(CUTTOOLINC)

JUNK=$(TARGETS) a.out

all: $(TARGETS)

gemm_cutlass_int.o: gemm_cutlass_int.cu
	$(NVCC) $< $(NVCCOPTs) $(OPTION) -o $@ -c -Xptxas -O3

conv_cutlass_int_NCHW.o: conv_cutlass_int.cu
	$(NVCC) $< $(NVCCOPTs) $(OPTION) -o $@ -c -Xptxas -O3 -DPIGEON_LAYOUT=\"NCHW\"

conv_cutlass_int_CHWN.o: conv_cutlass_int.cu
	$(NVCC) $< $(NVCCOPTs) $(OPTION) -o $@ -c -Xptxas -O3 -DPIGEON_LAYOUT=\"CHWN\"

clean:
	rm -rf $(JUNK)
