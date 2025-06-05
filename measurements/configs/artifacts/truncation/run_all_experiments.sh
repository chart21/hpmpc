#!/bin/bash
ITERATIONS=1 # replace with number of repeated executions you want to run
IP0=127.0.0.1
IP1=127.0.0.1
IP2=127.0.0.1
IP3=127.0.0.1
PID=all # replace with node id
REDUCED="NUM_INPUTS=1"
USE_CUDA_GEMM="0"

helpFunction()
{
   echo "Script to run all tests"
   echo -e "\t-p Party number"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-d IP address of player 3 (if ip matches player_id can be empty)"
   echo -e "\t-R Reduced settings for faster execution"
   echo -e "\t-G GPU Support"
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:D:L:R:i:g:" opt
do
   case "$opt" in
      p ) PID="$OPTARG" ;;
      a ) IP0="$OPTARG" ;;
      b ) IP1="$OPTARG" ;;
      c ) IP2="$OPTARG" ;;
      d ) IP3="$OPTARG" ;;
      D ) DATTYPE="$OPTARG" ;;
      L ) DATTYPES="$OPTARG" ;;
      R ) REDUCED="$OPTARG" ;;
      i ) ITERATIONS="$OPTARG" ;;
      g ) USE_CUDA_GEMM="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

echo "=====Starting all measurements with PID $PID, IP0 $IP0, IP1 $IP1, IP2 $IP2, IP3 $IP3, REDUCED $REDUCED, ITERATIONS $ITERATIONS, USE_CUDA_GEMM $USE_CUDA_GEMM====="

##4PC

echo "===Starting 4PC measurements for table 6==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table6 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 $REDUCED

echo "===Starting 4PC measurements for table 8==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table8 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 $REDUCED


echo "=====Finished all 4PC measurements====="
if [ $PID == 3 ]
then
    exit 0
fi

#3PC

echo "===Starting 3PC measurements for figure 14==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure14 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "===Starting 3PC measurements for figure 15==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure15 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "===Starting 3PC measurements for figure 16==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure16 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "===Starting 3PC measurements for figure 17==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure17 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "===Starting 3PC measurements for table 5==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table5 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "===Starting 3PC measurements for table 6==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table6 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "===Starting 3PC measurements for table 7==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table7 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

echo "=====Finished all 3PC measurements====="
