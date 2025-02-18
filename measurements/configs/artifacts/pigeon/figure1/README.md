# Instructions

Run all commands from the root directory of the repository.

## Setup

Run the following commands on each node.

```sh
export ITERATIONS=1 # replace with number of repeated executions you want to run
export IP0=10.10.94.2 # replace with your IPs
export IP1=10.10.94.3
export IP2=10.10.94.3
export IP3=10.10.94.3
export PID=0 # replace with node id
```

## Execution

Run the following commands on each node.

### Execute 3PC Experiments
This experiment requires a lot of processes and system RAM. If necessary, reduce the number of processes, e.g. by adding PROCESS_NUM=1 after --override.

```sh
python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5
```

### Execute 4PC Experiments
This experiment requires a lot of processes and system RAM. If necessary, reduce the number of processes, e.g. by adding PROCESS_NUM=1,2,4 after --override.

```sh
python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override PROTOCOL=9 
```

## Parse Results

Run the following commands on each node.

```sh
python3 measurements/parse_logs.py measurements/logs/
```

## Interpret Results

Open the csv files after parsing results. The data contains the number of bits per registers (DATTYPE), Number of Cores (Threads), and throughput in MBit/s.
