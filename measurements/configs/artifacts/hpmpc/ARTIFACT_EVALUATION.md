# Artifact Appendix

Paper title: **High Throughput Secure Multiparty Computation in Various Network Settings**

Artifacts HotCRP Id: **Submission #22 (2025.1)** 

Requested Badge: **Functional**

## Description
The artifact reproduces the experiments of all included figures and tables (except Table 1,2) of the paper. Table 1,2 only evaluate framework-independent or third-party framework measurements and are thus not covered. 
For each experiment, the artifact produces one or multiple csv files with measurement results that can be directly compared to the corresponding measurement point of a figure or an entry of a table in the paper.
The artifact includes an option to run the experiments with a reduced workload to test the functionality of the experiments and a full workload to reproduce the paper's results.
The reduced workload should complete within an hour on four multi-core machines in a distributed setup. It runs all tests with a reduced number of inputs and is therefore not comparable to runtimes and throughput achieved by the full results.
The full workload may take several hours to complete and requires high-performance hardware.
Different network settings (e.g., latency, bandwidth) are applied by the artifact using Linux traffic control.
All experiments can be executed using a single script and we provide a Dockerfile to run the experiments in a containerized environment.

### Security/Privacy Issues and Ethical Concerns (All badges)
We did not detect any security or privacy issues with the artifact or any ethical concerns.

## Basic Requirements 
The artifact can either be evaluated on a single machine or on multiple machines.
We recommend running the experiments on four distributed machines for more meaningful results.
Each machine should have identical hardware specifications.

### Hardware Requirements

We recommend a multi-core machine with at least 16 cores and the AES-NI instruction set for running the experiments.
For full reproducibility, please refer to the exact details in the `Run the experiments` section.

### Software Requirements
Each machine should come with a Debian-based operating system or Docker installed. For install instructions, please refer to the `Setup the environment` section.

### Estimated Time and Storage Consumption
The reduced workload should complete within an hour on multi-core machines while the full workload may take several hours to complete and requires high-performance hardware. For exact details, please refer to the `Run the experiments` section.


### Accessibility
Our artifact is publicly available on GitHub.

### Set up the environment 

The following instructions are for setting up the environment to run the experiments of the paper either on a Debian-based system or using Docker with an Ubuntu image.

#### Install dependencies and clone repository without Docker on Debian-based systems

Execute on each node to install dependencies (may require root privileges and running each command with sudo).
```bash
apt-get update && \
    apt-get install -y --no-install-recommends gcc-12 g++-12 libeigen3-dev libssl-dev git vim ca-certificates python3 jq bc build-essential iproute2 iperf && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100 
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

#### Hardware Support

You also need to export architecture-specific variables. As your machines may support different hardware instructions, you need to determine the highest supported register width that needs to be supported by all machines. The following line may help setting these variables automatically. 

```bash
export MAX_BITWIDTH=$(lscpu | grep -i flags | grep -q "avx512" && echo 512 || (lscpu | grep -i flags | grep -q "avx2" && echo 256 || (lscpu | grep -i flags | grep -q "sse" && echo 128 || echo 64)))
export SUPPORTED_BITWIDTHS=$(echo 1 8 16 32 64 128 256 512 | awk -v max="$MAX_BITWIDTH" '{for(i=1;i<=NF;i++) if($i<=max) printf $i (i<NF && $i<max?",":"")}')
```
When setting the environment variables manually, adhere to the following example:
```bash
export MAX_BITWIDTH=128 # Maximum supported bitwidth
export SUPPORTED_BITWIDTHS=1,8,16,32,64,128 # All supported bitwidth among 1,8,16,32,64,128,256,512 
```

#### Experiment Configuration

You can set the number of iterations for each experiment to receive multiple results for each measurement. 
```bash
export ITERATIONS=1 # Change to the desired number of iterations
```


### Testing the Environment
After setting up the environment, you can test the correctness of your setup by running the following command on each node.
If the nodes are able to connect and all tests pass, the environment is set up correctly.

```bash
make -j PARTY=$PID FUNCTION_IDENTIFIER=54 PROTOCOL=12 && scripts/run.sh -p $PID -n 4 -a $IP0 -b $IP1 -c $IP2 -d $IP3 
```


## Artifact Evaluation 

### Main Results and Claims
Our main results that should be supported by the artifact are the following:
- Result1: MPC can scale beyond a billion gates per second (more than 8Gbit/s throughput) on all protocols. This result can be verified by inspecting the throughput of the experiments for Table 6,7 but only on a sufficiently fast network, high-end hardware, and with the full workload.
- Result2: Our protocols achieve improved performance compared to existing protocols in various network settings. This result can be verified by comparing the throughput and runtimes of the experiments for Figure 10 between the different protocols. Requires running the full workload.

### Experiments 

#### Run the experiments (FUNCTIONALITY)

By default, the experiment script will run a single iteration and a heavily reduced workload to test the functionality of each experiment. 
All measurement points from the covered tables and figures are generated but with a smaller input size and thus lower throughput can be expected.
Also, the tested number of bits per register and the number of processes is reduced. Hence, one can expect all results from the different tables and figures but with lower throughput and runtime values and in some cases a smaller range of tested parameters per plot.

Run the following script on each node to execute the experiments.
Note that some experiments simulate different network settings by using Linux traffic control. Setting these may require root privileges and running the script below with sudo.
```bash
sudo ./measurements/configs/artifacts/hpmpc/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -L $SUPPORTED_BITWIDTHS -D $MAX_BITWIDTH # Run with sudo if necessary
```

#### Run the experiments (REPRODUCIBILITY)

To reproduce the results of the paper, we provide an option to run the full workload of each experiment by specifying -R "" in the script (not that the "" after -R is required). 
```bash
sudo ./measurements/configs/artifacts/hpmpc/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -L $SUPPORTED_BITWIDTHS -D $MAX_BITWIDTH -R "" # Run with sudo if necessary
```

Our nodes were configured as follows:
- CPU: AMD EPYC 7543 32-Core Processor
- RAM: 512GB
- Maximum Bitwidth: 256
- OS: Debian Trixie
- Network Links: 25 Gbit/s duplex link between all pairs of nodes without a switch (directly connected)
- Network Latency: 0.3ms

Note that the experiments may take a long time to complete. Significantly fewer cores or RAM on the nodes may crash the experiments.
Note also that the network topology is not equivalent to setting up servers on AWS with 25 Gbit/s links as AWS restricts the total bandwidth instead of the bandwidth per link to 25 Gbit/s.
Thus, we recommend a 100 Gbit/s link between the nodes when reproducing the results on AWS or cloud providers with similar policies.
The AWS instance setup that comes closest to our experiment setup is four m7a.32xlarge instances in the same availability zone.

#### Parse the results

To parse the log files to csv tables, run the following script after running the experiments.
```bash
python3 measurements/parse_logs.py measurements/logs
```
For each experiment, the script generates one or multiple csv files with the measurement data containing runtime or throughput values. The csv files are stored in the measurements/logs directory and are named after the experiment.

Protocols are represented by their respective IDs as shown in the table below. 

| Protocol | Adversary Model | Preprocessing | Supported Primitives |
| --- | --- | --- | --- |
| `1` Sharemind (3PC) | Semi-Honest | ✘ | Basic
| `2` Replicated (3PC) | Semi-Honest |✘  | Basic
| `3` ASTRA (3PC) | Semi-Honest | ✔ | Basic
| `4` ABY2 Dummy (2PC) | Semi-Honest | ✔ | Basic
| `5` Trio (3PC) | Semi-Honest | ✔ | All
| `6` Trusted Third Party (3PC) | Semi-Honest | ✘ | All
| `7` Trusted Third Party (4PC) | Semi-Honest | ✘ | All
| `8` Tetrad (4PC) | Malicious | ✔ | Basic
| `9` Fantastic Four (4PC) | Malicious | ✘ | Basic
| `10` Quad (4PC) | Malicious | ✘ | All
| `11` Quad: Het (4PC) | Malicious | ✔ | All
| `12` Quad (4PC) | Malicious | ✔ | All


The measurement data provided by the figures/tables corresponds to the columns of the .csv log files described below. Each experiment run produces a batch of one or multiple csv files. For instance, a batch may contain all 3PC experiments in one csv file while the 4PC experiments are saved in another csv file. Bandwidth and latency experiments produce a batch of csv files for each network configuration with a suffix to the filename indicating the network configuration, e.g. `_100Mbps`. Sometimes units in the csv files need to be converted to the desired unit in the paper (e.g. s to ms or Blocks into Million Blocks).

| Metric | Relevant Columns | Explanation |
| --- | --- | --- |
| TP (Gates/sec) | `TP_ONLINE_MAX(Mbit/s)` | The network throuhput in Mbit/s when running the experiment. Since all evaluations in the paper stating Gates/sec are based on boolean gates, the throughput in Mbit/s is equivalent to the throughput in 10^6 Gates/sec. `TP_ONLINE_MAX` measures the throughput based on the slowest process and is used for the results in the paper while `TP_ONLINE_AVG` measures the average throughput of all processes. |
| Runtime (s) | `ONLINE_MAX(s)` | The runtime of the experiment in seconds. `ONLINE_MAX` measures the runtime based on the slowest process (i.e. when all parallel processes are finished) and is used for the results in the paper while `ONLINE_AVG` measures the average runtime of all processes. |
| Bits per Register | `DATTYPE` | The number of bits per register used in the experiment. |
| Number of inputs | `NUM_INPUTS` | The number of inputs used in the experiment. It is set to 1 for FUNCTIONALITY experiments. |
| Bandwidth (Mbit/s) / Latency (ms) | - (Filename suffix) | Some experiments are evaluated with varying latencies/bandwidths. These experiments are evaluated once per specified network parameter. The resulting csv files contain a suffix indicating the network configuration, e.g. `_100Mbps` for 100Mbps bandwidth. |
| Blocks/s | `TP_ONLINE_MAX(OPs/s)` | The throughput in operations per second. For AES the number of blocks evaluated is equivalent to the number of OP/s. |
| Theoretical Limit (%) | - | The theoretical limit is calculated as the the achieved `TP_ONLINE_MAX(Mbit/s)` divided by the maximum bandwidth available on the network (e.g. 150Gbps resp. 300Gbps in case of a 25Gbps duplex link between each pair of three resp. four nodes). The theoretical limit needs to be calculated manually once the physical topology of the network is known. |
| Settings (e.g. MAN, WAN, Online) | - (Filename) | Some tables contain columns with different settings. The csv files contain a suffix indicating the setting, e.g. `_WAN.csv`. `PRE` indicates that a separate preprocessing phase is used and corresponds to the results marked with (Online), for instance in Table 7 and 8. |

#### Detailed Results

On successful completion of the experiments, the results can be found in the measurements/logs directory. The data can be interpreted as follows:

| Figure | x-Axis: Source | y-Axis: Source | Plot: Source |
| --- | --- | --- | --- |
| Figure 1 | Bandwidth (Mbit/s): `File suffix (e.g. _100Mbps)` | Runtime (ms): `ONLINE_MAX(s)` x10^3 | Multiplication: `FUNCTION_IDENTIFIER=2` <br /> Scalar Product: `FUNCTION_IDENTIFIER=7` <br /> Protocols: `PROTOCOL` |
| Figure 9 | Bits per Register: `DATTYPE` <br /> Number of Processes: `PROCESS_NUM` | Throughput (10^9 Gates/sec): `TP_ONLINE_MAX(Mbit/s)` / 1000 | Protocols: `PROTOCOL` |
| Figure 10 | Latency (ms): File suffix (e.g. `_8ms`) <br /> Bandwidth (Mbit/s): File suffix (e.g. `_100Mbps`) <br /> Input Size: `NUM_INPUTS` | Runtime (s): `ONLINE_MAX(s)` <br /> Throughput (10^9 Gates/sec): `TP_ONLINE_MAX(Mbit/s)` / 1000 | Protocols: `PROTOCOL` |
| Figure 29 | Latency (ms): File suffix (e.g. `_10ms`) | Runtime (s): `ONLINE_MAX(s)` | <br /> Multiplication: `FUNCTION_IDENTIFIER=8` <br /> Mult + Trunc: `FUNCTION_IDENTIFIER=9` <br /> A2B: `FUNCTION_IDENTIFIER=10` <br /> Bit2A: `FUNCTION_IDENTIFIER=11` <br /> Protocols: `PROTOCOL` |


|Table | Rows: Source | Columns: Source |
| --- | --- | --- |
| Table 6 | Protocols: `PROTOCOL` | Billion Gates/sec: `TP_ONLINE_MAX(Mbit/s)` / 1000 <br /> Theoretical Limit: `TP_ONLINE_MAX(Mbit/s)` / NETWORK_LIMIT(Mbit/s) |
| Table 7 | same as Table 6, (Online) refers to `PRE=1` | same as Table 6 |
| Table 8 | same as Table 7 | Million Blocks/sec: `ONLINE_MAX (OPs/s)`/10^6 <br /> Theoretical Limit: `TP_ONLINE_MAX(Mbit/s)` / NETWORK_LIMIT(Mbit/s) |
| Table 9 | Protocols: `PROTOCOL` | Runtime: `TP_ONLINE_MAX(s)` <br /> Lat: Filename contains `_ms` suffix <br /> Bdw: Filename contains `_Mbps` suffix <br /> Comp: Filename contains `_dot` |
| Table 10 | Protocols: `PROTOCOL` | Thousand Blocks/s: `TP_ONLINE_MAX(OPs/s)`/ 10^3 <br /> CMAN/WAN1/Mixed: Filename contains `_CMAN`/`_WAN1`/`_WAN2` suffix |


#### Automation of distributed tests with a Master Node

To run all tests from a single (external) master node that is not part of the computation servers and stream all outputs in the master node's terminal, you can fill in the `machines.json` file with the login credentials of 4 remote servers and run the following command on the seperate master node. This requires `pip install paramiko`.
```bash
cd hpmpc/measurements/configs/artifacts/hpmpc
python3 run_all_on_remote_servers.py
```
Alternatively, if you have tmux installed on the master node, you can run the following command for a cleaner terminal output in a 2x2 grid of the master node.
```bash
cd hpmpc/measurements/configs/artifacts/hpmpc
./run_with_tmux_grid.sh
```

#### TLDR

To run the experiments without further elaboration, complete the `Dependencies` and `Networking` sections and run the following commands on each machine. The results will be stored in the measurements/logs directory. The commands need to be executed on all machines simultaneously within 10 minutes to prevent connection timeouts.

```bash
export MAX_BITWIDTH=$(lscpu | grep -i flags | grep -q "avx512" && echo 512 || (lscpu | grep -i flags | grep -q "avx2" && echo 256 || (lscpu | grep -i flags | grep -q "sse" && echo 128 || echo 64)))
export SUPPORTED_BITWIDTHS=$(echo 1 8 16 32 64 128 256 512 | awk -v max="$MAX_BITWIDTH" '{for(i=1;i<=NF;i++) if($i<=max) printf $i (i<NF && $i<max?",":"")}')
export ITERATIONS=1
sudo ./measurements/configs/artifacts/hpmpc/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -L $SUPPORTED_BITWIDTHS -D $MAX_BITWIDTH # Run with sudo if necessary
python3 measurements/parse_logs.py measurements/logs
```



## Limitations 
Tables 1 and 2 of the paper do not evaluate our framework and are thus not covered by the artifact.

## Notes on Reusability 
Our repository allows users to develop new functions and experiments in a protocol-agnostic way that can be easily integrated into the existing framework.
We provide extensive documentation, tutorials, and examples on how to integrate new functions, protocols, and experiments into the framework.



