# Artifact Appendix

Paper title: **PIGEON: A High Throughput Framework for Private Inference of Neural Networks using Secure Multiparty Computation**

Artifacts HotCRP Id: **Submission #2 (2025.3)** 

Requested Badge: **Reproducable**

## Reviewer Instructions

We set up machines and provided a `machines.json` file with login information in the submission portal. All servers already have the neccessary dependencies installed. To run all experiments, clone the `HPMPC` repository to your PC, copy the `machines.json` to the `measurements/configs/artifacts/pigeon` directory, and jump to the [Automation of distributed tests with a Master Node](#automation-of-distributed-tests-with-a-master-node) section for instructions to execute all experiments. 

## Description
The artifact reproduces the experiments of all included figures and tables (exluding evaluation of third-party frameworks and evaluation of subroutines) of the paper. Table 1 only evaluates a third-party framework and is thus not covered. 
For each experiment, the artifact produces one or multiple csv files with measurement results that can be directly compared to the corresponding measurement point of a figure or an entry of a table in the paper.
The artifact includes an option to run the experiments with a reduced workload to test the functionality of the experiments and a full workload to reproduce the paper's results.
The reduced workload should complete within two hours on four multi-core machines in a distributed setup. It runs all tests with a reduced number of inputs and is therefore not comparable to runtimes and throughput achieved by the full results.
The full workload should also complete within two hours but requires high-performance hardware (32 cores, AVX2, 512GB RAM, ideally an NVIDIA GPU with at least 24GB VRAM).
All experiments can be executed using a single script and we provide a Dockerfile to run the experiments in a containerized environment.

### Security/Privacy Issues and Ethical Concerns (All badges)
We did not detect any security or privacy issues with the artifact or any ethical concerns.

## Basic Requirements 
The artifact can either be evaluated on a single machine or on multiple machines.
We recommend running the experiments on four distributed machines for more meaningful results.
Each machine should have identical hardware specifications.

### Hardware Requirements

Each node must support AVX512 to run the scripts without manual modifications.
We recommend four multi-core machines with at least 16 cores and 64GB RAM for running the reduced workload.
We recommend four multi-core machines with at least 32 cores, 512GB RAM, and a 24GB GPU for running the full workload.
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

- Result1: Increasing the utilized bits per register with vectorization and bitslicing improves network throughput significantly (Figure 1).
- Result2: All relevant tables and figures from the paper can be re-evaluated with the artifact.

### Experiments 

#### Run the experiments (FUNCTIONALITY)

By default, the experiment script will run a single iteration and a heavily reduced workload to test the functionality of each experiment. 
All measurement points from the covered tables and figures are generated but with a smaller input size.
Also, the tested number of bits per register and the number of processes is reduced. Hence, one can expect all results from the different tables and figures but with lower throughput and runtime values and in some cases a smaller range of tested parameters per plot.

Run the following script on each node simultaneously to execute the experiments. If your machines have GPUs, you can set the `-g 2` flag to also run GPU-based experiments. 
```bash
./measurements/configs/artifacts/pigeon/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS  
```

#### Run the experiments (REPRODUCIBILITY)

To reproduce the results of the paper, we provide an option to run the full workload of each experiment by specifying -R "" in the script (not that the "" after -R is required). 
```bash
sudo ./measurements/configs/artifacts/hpmpc/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -R "" 
```

Our nodes were configured as follows:
Four AWS G6 16xlarge instances with the following properties:

    - CPU: AMD EPYC 7543 32-Core Processor
    - GPU: NVIDIA L4 GPU, 32GB
    - RAM: 512GB
    - Maximum Bitwidth: 256
    - OS: Ubuntu LTS 22.04
    - Network Links: 25 Gbit/s duplex link between all pairs of nodes without a switch (directly connected)
    - Network Latency: 0.3ms

Note that fewer cores or RAM on the nodes may crash the experiments.
For some 64-bit tests we utilized CPU-based machines with AVX-512 support such as R7a instances due to the missing AVX-512 support of instances with a GPU.

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
| TP (Gates/sec) | `TP_ONLINE_MAX(Mbit/s)` | The network throughput in Mbit/s when running the experiment. |
| Runtime (s) | `ONLINE_MAX(s)` | The runtime of the experiment in seconds. `ONLINE_MAX` measures the runtime based on the slowest process (i.e. when all parallel processes are finished) and is used for the results in the paper while `ONLINE_AVG` measures the average runtime of all processes. |
| Bits per Register | `DATTYPE` | The number of bits per register used in the experiment. |
| Number of inputs | `NUM_INPUTS` | The number of inputs used in the experiment. It is set to 1 for FUNCTIONALITY experiments. |
| Blocks/s | `TP_ONLINE_MAX(OPs/s)` | The throughput in operations per second. |

#### Detailed Results

On successful completion of the experiments, the results can be found in the measurements/logs directory. The data can be interpreted as follows:

##### Figures

| Figure | x-Axis: Source | y-Axis: Source | Plot: Source |
| --- | --- | --- | --- |
| Figure 1 | a,c) Bits per register: `DATTYPE` <br /> b,d) Number of Threads: `PROCESS_NUM` | Throughput: `TP_ONLINE_MAX(Mbit/s)`/1000 | 3PC: `PROTOCOL=5` <br /> 4PC: `PROTOCOL=12` |
| Figure 3 | Threads: `PROCESS_NUM` | Throughput (Gbit/s): `TP_ONLINE_MAX(Mbit/s)` / 1000 | 3PC: `PROTOCOL=5`<br />  4PC: `PROTOCOL=12` <br />  Naive: `FUSE_DOT=0`, `INTERLEAVE_COMM=0`<br /> MPC-friendly: `FUSE-DOT=1`, `INTERLEAVE_COMM=1`, <br /> GPU: `USE_CUDA_GEMM=2` |
| Figure 6 | Input Size: `NUM_INPUTS` | Runtime (ms): `ONLINE_MAX(s)` * 1000 | 3PC: `PROTOCOL=5`<br /> 4PC: `PROTOCOL=12`<br /> Batch Size: `DATTYPE/(BITLENGTH*SPLITROLES_FACTOR*PROCESS_NUM`) |


##### Tables

|Table | Rows: Source | Columns: Source |
| --- | --- | --- |
| Table 4,7,8,9 | Setting: `PROTOCOL`<br /> PIGEON CPU: `USE_CUDA_GEMM=0` <br /> PIGEON GPU (if available): `USE_CUDA_GEMM=2` | AlexNet (CIFAR-10): `FUNCTION_IDENTIFIER=180`<br /> ResNet18 (CIFAR-10): `FUNCTION_IDENTIFIER=170`<br /> ResNet50 (CIFAR-10): `FUNCTION_IDENTIFIER=171`<br /> VGG-16 (CIFAR-10): `FUNCTION_IDENTIFIER=174`<br /> ResNet18 (ImageNet): `FUNCTION_IDENTIFIER=175`<br /> ResNet50 (ImageNet): `FUNCTION_IDENTIFIER=176`<br /> VGG-16 (ImageNet) `FUNCTION_IDENTIFIER=179`<br /> Throughput (Images/s): `Throughput(Op/s)` |
| Table 5 | Total Runtime (s): `ONLINE_MAX(s)`<br /> Total Gbps: `TP_ONLINE_MAX(Mbit/s)` / 1000<br /> Total GB: `ONLINE_SENT(MB)`/1000 | VGG-16: `FUNCTION_IDENTIFIER=179`<br /> ResNet152: `FUNCTION_IDENTIFIER=178` | 
Table 6 | PPA: `FUNCTION_IDENTIFIER` of CNN<br /> RCA: `FUNCTION_IDENTIFIER` of CNN subtracted by `100` <br /> RCA_8: `COMPRESS`=1<br /> ON: `PRE=1`<br /> PIGEON CPU: `USE_CUDA_GEMM=0`<br /> PIGEON GPU (if available): `USE_CUDA_GEMM=2` | VGG-16 (CIFAR-10): `FUNCTION_IDENTIFIER=174`<br /> ResNet50 (ImageNet): `FUNCTION_IDENTIFIER=176`<br /> VGG-16 (ImageNet) `FUNCTION_IDENTIFIER=179`<br /> Batch Size: `DATTYPE/BITLENGTH*SPLITROLES_FACTOR*PROCESS_NUM` <br />  Throughput (Images/s): `Throughput(Op/s)` |


## Automation of distributed tests with a Master Node

To run all tests from a single (external) master node that is not part of the computation servers and stream all outputs in the master node's terminal, you can fill in the `machines.json` file with the login credentials of 4 remote servers and run the following command on the seperate master node. This requires `pip install paramiko`. The experiment results will be stored as csv files on each node locally in the `hpmpc/measurements/logs` directory and also on the master node in the `hpmpc/measurements/logs/node_$PID/` directory. For GPU support, additionally set `-g 2` in the commands below.

### For Functionality 

```bash
cd hpmpc/measurements/configs/artifacts/pigeon
python3 run_all_on_remote_servers.py -p all
```
Alternatively, if you have tmux installed on the master node, you can run the following command for a cleaner terminal output in a 2x2 grid of the master node.
```bash
cd hpmpc/measurements/configs/artifacts/pigeon
./run_with_tmux_grid.sh
```

### For Reproducibility

```bash
cd hpmpc/measurements/configs/artifacts/pigeon
python3 run_all_on_remote_servers.py -p all -R ""
```
Alternatively, if you have tmux installed on the master node, you can run the following command for a cleaner terminal output in a 2x2 grid of the master node.
```bash
cd hpmpc/measurements/configs/artifacts/pigeon
./run_with_tmux_grid.sh -R ""
```


## Limitations 

Results of third-party frameworks are not included in the artifact.



