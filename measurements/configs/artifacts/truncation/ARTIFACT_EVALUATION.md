# Artifact Appendix

Paper title: **SoK: Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets**

Artifacts HotCRP Id: **Submission #6 (2025.4)** 

Requested Badge: **Reproducible**

## Reviewer Instructions

We set up machines and provided a `machines.json` file with login information in the submission portal. All servers already have the neccessary dependencies installed. To run all experiments, clone the `HPMPC` repository to your PC, copy the `machines.json` to the `measurements/configs/artifacts/truncation` directory, and jump to the [Automation of distributed tests with a Master Node](#automation-of-distributed-tests-with-a-master-node) section for instructions to execute all experiments. 

## Description
The artifact reproduces the experiments of all included figures and tables in the main body of the paper.  
For each experiment, the artifact produces one or multiple csv files with measurement results that can be directly compared to the corresponding measurement point of a figure or an entry of a table in the paper.
For experiments in specific network environments, bandwidth and latency are simulated using Linux traffic control (tc).
The artifact includes an option to run the experiments with a reduced workload to test the functionality of the experiments and a full workload to reproduce the paper's results.
The reduced workload should complete within two hours on four multi-core machines in a distributed setup. It runs all tests with a reduced number of inputs and is therefore not comparable to runtimes and accuracy achieved by the full results.
The full workload should also complete within two hours but requires high-performance hardware (16 cores, AVX2, 64GB RAM).
All experiments can be executed using a single script and we provide a Dockerfile to run the experiments in a containerized environment.

### Security/Privacy Issues and Ethical Concerns (All badges)
We did not detect any security or privacy issues with the artifact or any ethical concerns.

## Basic Requirements 
The artifact can either be evaluated on a single machine or on multiple machines.
We recommend running the experiments on four distributed machines for more meaningful results.
Each machine should have identical hardware specifications.

### Hardware Requirements

Each node must support AVX512 to run the scripts without manual modifications.
We recommend four multi-core machines with at least 4 cores and 16GB RAM for running the reduced workload.
We recommend four multi-core machines with at least 16 cores, 64GB RAM, for running the full workload.
For full reproducibility, please refer to the exact details in the `Run the experiments` section.

### Software Requirements
Each machine should come with a Debian-based operating system or Docker installed. For install instructions, please refer to the `Setup the environment` section.

### Estimated Time and Storage Consumption
All workload should complete within two hours in a LAN setting with the specified hardware requirements. For exact details, please refer to the `Run the experiments` section.


### Accessibility
Our artifact is publicly available on GitHub.

### Set up the environment 

The following instructions are for setting up the environment to run the experiments of the paper either on a Debian-based system or using Docker with an Ubuntu image.

#### Install dependencies and clone repository without Docker on Debian-based systems

Execute on each node to install dependencies (may require root privileges and running each command with sudo).
```bash
sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends gcc-12 g++-12 libeigen3-dev libssl-dev git vim ca-certificates python3 jq bc build-essential iproute2 iperf && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100 
```

Clone the repository and its submodules.
```bash
git clone https://github.com/chart21/hpmpc && \
    cd hpmpc && \
    git submodule update --init --recursive
```

#### Alternatively: Build an Image and run the container with Docker

```bash
#Run with Docker
docker build -t hpmpc .
#Run each command in different terminals or different machines
 docker run -it --network host --cap-add=NET_ADMIN --name p0 hpmpc
 docker run -it --network host --cap-add=NET_ADMIN --name p1 hpmpc
 docker run -it --network host --cap-add=NET_ADMIN --name p2 hpmpc
 docker run -it --network host --cap-add=NET_ADMIN --name p3 hpmpc
```

#### Download pretrained model weights and datasets
```bash
cd nn/Pygeon
python download_pretrained.py all
cd ../..
```

#### Optional: Set up GPU Support

To run all GPU-based experiments on the CPU, this step can be omitted.

```bash
# Dependencies for GPU acceleration
git clone https://github.com/NVIDIA/cutlass.git

# Compile standalone executable for GPU acceleration
cd core/cuda
# Replace with your GPU architecture, nvcc path, and CUTLASS path:
make -j arch=sm_89 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/home/user/cutlass
cd ../..
```

### Set environment variables


#### Networking
We recommend setting the following environment variables on each node. Replace the IP addresses with the actual IP addresses of the nodes. The PID has to be set to the ID of the node (0, 1, 2, or 3) for each machine/container individually. Use PID=all for a single-machine setup.
```bash
export IP0=127.0.0.1 
export IP1=127.0.0.1
export IP2=127.0.0.1
export IP3=127.0.0.1
export PID=0
```
#### Experiment Configuration

You can set the number of iterations for each experiment to receive multiple results for each measurement. 
```bash
export ITERATIONS=1 # Change to the desired number of iterations
```


### Testing the Environment
After setting up the environment, you can test the correctness of your setup by running the following command on each node simultaneously.
If the nodes are able to connect and all tests pass, the environment is set up correctly.

```bash
make -j PARTY=$PID FUNCTION_IDENTIFIER=54 PROTOCOL=12 && scripts/run.sh -p $PID -n 4 -a $IP0 -b $IP1 -c $IP2 -d $IP3 
```

To test GPU support, run the following command on each node simultaneously.

```bash
make -j USE_CUDA_GEMM=2 PARTY=$PID FUNCTION_IDENTIFIER=57 PROTOCOL=12 && scripts/run.sh -p $PID -n 4 -a $IP0 -b $IP1 -c $IP2 -d $IP3 
```

## Artifact Evaluation 

### Main Results and Claims
Our main results that should be supported by the artifact are the following:

- Result 1: The accuracy results of the paper can be reproduced with the artifact.
- Result 2: The runtime results of the paper can be reproduced with the artifact.

### Experiments 

#### Run the experiments (FUNCTIONALITY)

By default, the experiment script will run at a reduced workload. 
All measurement points from the covered tables and figures are generated but with a smaller input size, i.e. the number of images is reduced.
Hence, one can expect slightly different results for accuracy and runtime compared to the results in the paper.

Run the following script on each node simultaneously to execute the experiments. If your machines have GPUs, you can set the `-g 2` flag to also run GPU-based experiments. 
```bash
./measurements/configs/artifacts/truncation/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS  
```

#### Run the experiments (REPRODUCIBILITY)

To reproduce the results of the paper, we provide an option to run the full workload of each experiment by specifying -R "" in the script (not that the "" after -R is required). 
```bash
sudo ./measurements/configs/artifacts/truncation/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -R "" 
```

Our nodes were configured as follows:

- CPU: AMD EPYC 7543 32-Core Processor
- RAM: 512GB
- Maximum Bitwidth: 256
- OS: Debian Trixie
- Network Links: 25 Gbit/s duplex link between all pairs of nodes without a switch (directly connected)
- Network Latency: 0.3ms


### Parse the results

To parse the log files to csv tables, run the following script after running the experiments.
```bash
python3 measurements/parse_logs.py measurements/logs
```
For each experiment, the script generates one or multiple csv files with the measurement data containing runtime, throughput, or accuracy values. The csv files are stored in the measurements/logs directory and are named after the experiment.

Protocols are represented by their respective IDs as shown in the table below. 

| Protocol | Adversary Model | Preprocessing | Supported Primitives |
| --- | --- | --- | --- |
| `5` Trio (3PC) | Semi-Honest | ✔ | All
| `12` Quad (4PC) | Malicious | ✔ | All


The measurement data provided by the figures/tables corresponds to the columns of the .csv log files described below. 

| Metric | Relevant Columns | Explanation |
| --- | --- | --- |
| Runtime (s) | `ONLINE_MAX(s)` | The runtime of the experiment in seconds. `ONLINE_MAX` measures the runtime based on the slowest process (i.e. when all parallel processes are finished) and is used for the results in the paper while `ONLINE_AVG` measures the average runtime of all processes. |
| Accuracy (%) | `ACCURACY(%)` | The accuracy achieved in the experiment. |
| Truncation Primitive | `TRUNC_APPROACH` | TS_L: `TRUNC_APPROACH=0`, TS_1: `TRUNC_APPROACH=1`, TE_0: `TRUNC_APPROACH=2`, TE_1: `TRUNC_APPROACH=3`, TS_Mix: `TRUNC_APPROACH=4` |
| Model Architecture | `FUNCTION_IDENTIFIER` | The identifier of the model architecture used in the experiment. We use the following mapping:
`182` - LeNet on MNIST <br />
`171` - ResNet50 on CIFAR-10 <br />
`174` - VGG-16 on CIFAR-10 <br />
`186` - VGG-16 on ImageNet (as specified by PyTorch) <br />
| Number of fractional bits | `FRACTIONAL` | The number of fractional bits used in fixed point representation. |
| Optimizations | Flags indicate whether certain optimizations such as `TRUNC_DELAYED` or `AVG_OPT` are active (1) or not (0/missing) |


## Automation of distributed tests with a Master Node

To run all tests from a single (external) master node that is not part of the computation servers and stream all outputs in the master node's terminal, you can fill in the `machines.json` file with the login credentials of 4 remote servers and run the following command on the seperate master node. This requires `pip install paramiko`. The experiment results will be stored as csv files on each node locally in the `hpmpc/measurements/logs` directory and also on the master node in the `hpmpc/measurements/logs/node_$PID/` directory. For GPU support, additionally set `-g 2` in the commands below.

### For Functionality 

```bash
cd hpmpc/measurements/configs/artifacts/truncation
python3 run_all_on_remote_servers.py -p all
```
Alternatively, if you have tmux installed on the master node, you can run the following command for a cleaner terminal output in a 2x2 grid of the master node.
```bash
cd hpmpc/measurements/configs/artifacts/truncation
./run_with_tmux_grid.sh
```

### For Reproducibility

```bash
cd hpmpc/measurements/configs/artifacts/truncation
python3 run_all_on_remote_servers.py -p all -R ""
```
Alternatively, if you have tmux installed on the master node, you can run the following command for a cleaner terminal output in a 2x2 grid of the master node.
For the optimized version on weaker hardware with reduced parallelization factor and batch sizes (e.g. 16 cores, 64GB RAM), you can add the `-O 16Core_VMS` flag to the command.

```bash
cd hpmpc/measurements/configs/artifacts/truncation
./run_with_tmux_grid.sh -R "\"\""  # Add [-O 16Core_VMS] for weaker hardware
```





