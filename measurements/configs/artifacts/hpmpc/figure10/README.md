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
export DATTYPE=256 # replace with highest DATTYPE supported by your hardware
epxort DATTYPES=1,8,16,32,64,128,256 # replace with all DATTYPES supported by your hardware
```


## Execution

Run the following commands on each node.

### Execute 3PC Experiments (1)
```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10a_aes_latency.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPES PROTOCOL=2,5
```

### Execute 4PC Experiments (1)
```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10b_aes_latency.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPES PROTOCOL=9,10,12
```

### Execute 4PC Experiments (2)

This experiment requires a lot of processes and system RAM. If necessary, reduce the number of inputs and num processes, e.g. by adding PROCESS_NUM=4 NUM_INPUTS=1000 after --override.

```sh
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10c_vector_matrix.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10b_mult_throughput.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPE
```

## Parse Results

Run the following commands on each node.

```sh
python3 measurements/parse_logs.py measurements/logs/
```

## Interpret Results

Open the csv files after parsing results. The achieved throughput should be contained in the table.
