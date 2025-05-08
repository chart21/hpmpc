#!/bin/bash
ITERATIONS=1 # replace with number of repeated executions you want to run
IP0=127.0.0.1
IP1=127.0.0.1
IP2=127.0.0.1
IP3=127.0.0.1
PID=all # replace with node id
REDUCED="PROCESS_NUM=1 SPLITROLES=0"
REDUCED_PROCESS_NUM="PROCESS_NUM=1,2,4,8"
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

echo "===Starting 4PC measurements for figure 1==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 $REDUCED_PROCESS_NUM

echo "===Starting 4PC measurements for figure 3==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure3/figure3CPU.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 USE_CUDA_GEMM=$USE_CUDA_GEMM $REDUCED_PROCESS_NUM

if [ $USE_CUDA_GEMM != "0" ]
then
    python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure3/figure3GPU.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=5 USE_CUDA_GEMM=2 $REDUCED_PROCESS_NUM
fi

echo "===Starting 4PC measurements for figure 6==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 PROCESS_NUM=1 SPLITROLES=0,3 $REDUCED NUM_INPUTS=10000,100000,1000000

echo "===Starting 4PC measurements for table 4==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table4/table4.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 PROCESS_NUM=1 SPLITROLES=3 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=180,171

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table4/table4.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 PROCESS_NUM=1 SPLITROLES=0 PROCESS_NUM=8 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=176,179

echo "===Starting 4PC measurements for table 5==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table5/table5.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 SPLITROLES=0 PROCESS_NUM=4 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=178

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table5/table5.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 SPLITROLES=0 PROCESS_NUM=8 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=179

echo "===Starting 4PC measurements for table 7==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table7/table7.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 SPLITROLES=3 PROCESS_NUM=1 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=180,171,174

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table7/table7.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 SPLITROLES=0 PROCESS_NUM=8 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=175

# python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table7/table7.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 SPLITROLES=0 PROCESS_NUM=1 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=179


echo "===Starting 4PC measurements for table 8==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table8 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 PROCESS_NUM=1 USE_CUDA_GEMM=$USE_CUDA_GEMM 

echo "===Starting 4PC measurements for table 9==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table9 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 PROCESS_NUM=1 USE_CUDA_GEMM=$USE_CUDA_GEMM 


echo "=====Finished all 4PC measurements====="
if [ $PID == 3 ]
then
    exit 0
fi

#3PC

echo "===Starting 3PC measurements for figure 1==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED_PROCESS_NUM

echo "===Starting 3PC measurements for figure 3==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure3/figure3CPU.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=5 USE_CUDA_GEMM=$USE_CUDA_GEMM $REDUCED_PROCESS_NUM

if [ $USE_CUDA_GEMM != "0" ]
then
    python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure3/figure3GPU.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=5 USE_CUDA_GEMM=2 $REDUCED_PROCESS_NUM
fi


echo "===Starting 3PC measurements for figure 6==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/figure6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=5 SPLITROLES=0,1 $REDUCED NUM_INPUTS=10000,100000,1000000

echo "===Starting 3PC measurements for table 4==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table4/table4.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PROCESS_NUM=4 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=180,171,174

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table4/table4.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PROCESS_NUM=1 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=176,179

echo "===Starting 3PC measurements for table 5==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table5/table5.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PROCESS_NUM=1 PRE=0 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM 

echo "===Starting 3PC measurements for table 6==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table6/table6_cifar.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PROCESS_NUM=4 PRE=0 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table6/table6_imagenet.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PRE=0 PROCESS_NUM=1 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM

echo "===Starting 3PC measurements for table 7==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table7/table7.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PROCESS_NUM=4 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=180,171,174

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table7/table7.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=1 PROCESS_NUM=1 $REDUCED USE_CUDA_GEMM=$USE_CUDA_GEMM FUNCTION_IDENTIFIER=175,179

echo "===Starting 3PC measurements for table 8==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table8/table8.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 SPLITROLES=0 USE_CUDA_GEMM=$USE_CUDA_GEMM 

echo "===Starting 3PC measurements for table 9==="

python3 measurements/run_config.py measurements/configs/artifacts/pigeon/table9 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=5 USE_CUDA_GEMM=$USE_CUDA_GEMM 

echo "=====Finished all 3PC measurements====="
