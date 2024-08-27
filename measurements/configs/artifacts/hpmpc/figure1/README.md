# Instructions

Run all commands from the root directory of the repository.

## Setup

Run the following commands on each node.

```sh
export ITERATIONS=1 # replace with number of repeated executions you want to run
export P0=10.10.94.2 # replace with your IPs
export IP1=10.10.94.3
export IP2=10.10.94.3
export IP3=10.10.94.3
export PID=0 # replace with node id
export DATTYPE=256 # replace with highest DATTYPE supported by your hardware
```

## Network Shaping

Make sure to apply the bandwidths from figure 1 for each run.

## Execution

Run the following commands on each node.

### Execute 3PC Experiments
```sh
measurements/run_config.sh measurements/configs/artifacts/hpmpc/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2 DATTYPE=$DATTYPE
```

### Execute 4PC Experiments
```sh
measurements/run_config.sh measurements/configs/artifacts/hpmpc/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override PROTOCOL=9 DATTYPE=$DATTYPE
```

## Parse Results

Run the following commands on each node.

```sh
measurements/parse_logs.sh measurements/logs/
```

## Interpret Results

Open the csv files after parsing results. Divide Runtime by PROCESS_NUM to get the time taken per process.
