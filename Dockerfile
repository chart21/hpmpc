FROM ubuntu:24.04

# Install necessary packages 
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc-12 g++-12 libeigen3-dev libssl-dev git vim ca-certificates python3 jq bc build-essential iproute2 iperf && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100 && \
    rm -rf /var/lib/apt/lists/*

# Clone repositories and set up repository
RUN git clone https://github.com/chart21/hpmpc && \
    cd hpmpc && \
    git submodule update --init --recursive

WORKDIR /hpmpc

ENTRYPOINT ["/bin/bash"]
