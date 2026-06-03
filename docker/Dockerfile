# Base image — CUDA devel includes nvcc, required for GPU acceleration (USE_CUDA_GEMM)
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-12 g++-12 \
        gcc-13 g++-13 \
        build-essential \
        cmake \
        ninja-build \
        libeigen3-dev \
        libssl-dev \
        libgmp-dev \
        libntl-dev \
        git \
        curl \
        wget \
        vim \
        ca-certificates \
        python3 \
        python3-pip \
        jq \
        bc \
        iproute2 \
        iperf \
        iperf3 \
        iputils-ping && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100 && \
    rm -rf /var/lib/apt/lists/*

# Python packages for PIGEON/Pygeon and measurements
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision gdown numpy

# Clone CUTLASS for GPU-accelerated matrix multiplication and convolutions
# H100 architecture: sm_90 — set arch=sm_90 when compiling core/cuda
RUN git clone --depth 1 https://github.com/NVIDIA/cutlass.git /cutlass

ENV CUDA_PATH=/usr/local/cuda
ENV CUTLASS_PATH=/cutlass

# GPU architecture for ConvTriple's troy-nova HE library (H100 = 90)
# Used by nn/ConvTriple/deps.sh -gpu
ENV GPU_ARCHITECTURE=90

# The hpmpc repo is expected to be bind-mounted at /hpmpc at runtime:
#   docker run -v /path/to/hpmpc:/hpmpc ...
WORKDIR /hpmpc

ENTRYPOINT ["/bin/bash"]
