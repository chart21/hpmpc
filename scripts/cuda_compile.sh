CUTLASS_PATH="/home/christopher/workspace/cutlass" # replace with your cutlass path, set up with official CUTLASS quick start guide
CUDALIB="/opt/cuda/lib64" # replace with your cuda library path
CUTLASSLIB="${CUTLASS_PATH}/build/tools/library/"
#loop over all .o files in the current directory
for p in *.o; do
    nvcc -L ${CUTLASSLIB} -L ${CUDALIB} ./cuda/gemm_cutlass_int.o $p -o ${p}-cuda
    mv $p-cuda $p
done
