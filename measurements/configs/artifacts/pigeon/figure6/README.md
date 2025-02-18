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
```


## Execution

Run the following commands on each node.

This experiment requires a lot of processes and system RAM. If necessary, reduce the number of processes, e.g. and inputs, e.g. by adding PROCESS_NUM=1 and NUM_INPUTS=10,100 after --override.


```sh
python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override PROTOCOL=5 SPLITROLES=0,1
```

### Execute 4PC Experiments


```sh
python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override PROTOCOL=12 SPLITROLES=0,3
```

## Parse Results

Run the following commands on each node.

```sh
python3 measurements/parse_logs.py measurements/logs/
```

## Interpret Results

Open the csv files after parsing results. The achieved throughput should be contained in the table.
