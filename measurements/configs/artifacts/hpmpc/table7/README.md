# Instructions

Run all commands from the root directory of the repository.

## Setup

Run the following commands on each node.

```sh
export ITERATIONS=1 # replace with number of repeated executions you want to run
export IP0=10.10.94.2 # replace with your IPs
export IP1=10.10.94.3
export IP2=10.10.94.4
export IP3=10.10.94.5
export PID=0 # replace with node id
export DATTYPE=256 # replace with highest DATTYPE supported by your hardware
```

## Execution

Run the following commands on each node.

### Execute 3PC Experiments
This experiment requires a lot of processes and system RAM. If necessary, reduce the number of processes or input size, e.g. by adding PROCESS_NUM=1 NUM_INPUTS=10000 after --override.

```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_3PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_3PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE
```

### Execute 4PC Experiments
This experiment requires a lot of processes and system RAM. If necessary, reduce the number of processes or input size, e.g. by adding PROCESS_NUM=4 NUM_INPUTS=10000 after --override.

```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_4PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_4PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE
```

## Parse Results

Run the following commands on each node.

```sh
python3 measurements/parse_logs.py measurements/logs/
```

## Interpret Results

Open the csv files after parsing results to obtain throughput.
