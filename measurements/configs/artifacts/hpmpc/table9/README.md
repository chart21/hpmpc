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

## Network Shaping

Make sure to apply the bandwidths/latencies from table 9 for each run.

## Execution

Run the following commands on each node.

### Execute Experiments
The experiments except aes_lat require a lot of processes and system RAM. If necessary, reduce the number of processes or input size, e.g. by adding PROCESS_NUM=1 NUM_INPUTS=1000 after --override.

#### Execute 3PC Experiments

```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
```


#### Execute 4PC Experiments

```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE
```



## Parse Results

Run the following commands on each node.

```sh
python3 measurements/parse_logs.py measurements/logs/
```

## Interpret Results

Open the csv files after parsing results to obtain runtime.
