#SET YOUR OPTIONS HERE
NVCC=/opt/cuda/bin/nvcc # replace with which nvcc
NVCCOPTs= -arch=sm_52 --expt-relaxed-constexpr  -std=c++20 # replace with your architecture
CUTLASS_PATH=/home/christopher/workspace/cutlass# replace with your cutlass path, set up with official CUTLASS quick start guide
CUDALIB=/opt/cuda/lib64 # replace with your cuda library path


TARGETS= gemm_cutlass_int.o

CUTLASSINC=$(CUTLASS_PATH)/include
CUTTOOLINC=$(CUTLASS_PATH)/tools/util/include
CUTLASSLIB=$(CUTLASS_PATH)/build/tools/library/


# OPTION= -L$(CUDALIB)  -Iinclude/  -I$(CUTLASSINC) -I$(CUTTOOLINC) -L$(CUTLASSLIB)
OPTION= -Iinclude/  -I$(CUTLASSINC) -I$(CUTTOOLINC) 

JUNK=$(TARGETS) a.out


all: $(TARGETS)

%.o: %.cu
	$(NVCC) $< $(NVCCOPTs)  $(OPTION) -o $@ -c

clean:
	rm -rf $(JUNK)
