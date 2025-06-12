#!/bin/bash
ITERATIONS=1 # replace with number of repeated executions you want to run
IP0=127.0.0.1
IP1=127.0.0.1
IP2=127.0.0.1
IP3=127.0.0.1
PID=all # replace with node id
REDUCED="NUM_INPUTS=1"
IMAGENET=""
USE_CUDA_GEMM="0"
EXPERIMENTS="all"

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
   echo -e "\t-i Number of iterations to run"
   echo -e "\t-E Experiments to run (default: all, options: Table6, Table8, Figure14, Figure15, Figure16, Table5)"
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:D:L:R:i:g:E:" opt
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
      E ) EXPERIMENTS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

export MODEL_DIR=nn/Pygeon/models/pretrained
export DATA_DIR=nn/Pygeon/data/datasets

echo "=====Starting all measurements with PID $PID, IP0 $IP0, IP1 $IP1, IP2 $IP2, IP3 $IP3, REDUCED $REDUCED, ITERATIONS $ITERATIONS====="


##4PC

if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Table6" ]; then

echo "===Starting 4PC measurements for table 6==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table6 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 $REDUCED

fi


if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Table8" ]; then

echo "===Starting 4PC measurements for table 8 with 1Gbps bandwidth and 2ms latency==="


measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -B 1000 -L 2

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table8 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 $REDUCED

measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1 #Reset network shaping

fi


echo "=====Finished all 4PC measurements====="
if [ $PID == 3 ]
then
    exit 0
fi

#3PC

if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Figure14" ]; then

echo "===Starting 3PC measurements for figure 14==="

export MODEL_FILE=Cifar_adam_001/ResNet50_avg_CIFAR-10_standard_best.bin
export SAMPLES_FILE=CIFAR-10_standard_test_images.bin
export LABELS_FILE=CIFAR-10_standard_test_labels.bin

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure14/figure14_32bit.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

export MODEL_FILE=adam_001_wd/ResNet50_avg_AdamW_d05_wd003_lr0001_ep100_acc74_35.bin

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure14/figure14_weight_decay.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

fi


if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Figure15" ]; then

echo "===Starting 3PC measurements for figure 15==="
        
export MODEL_FILE=MNIST_LeNet5/LeNet5_MNIST_standard_best.bin
export SAMPLES_FILE=MNIST_standard_test_images.bin
export LABELS_FILE=MNIST_standard_test_labels.bin


python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure15 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

fi

if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Figure16" ]; then


echo "===Starting 3PC measurements for figure 16==="

export MODEL_FILE=Cifar_adam_001/ResNet50_avg_CIFAR-10_standard_best.bin
export SAMPLES_FILE=CIFAR-10_standard_test_images.bin
export LABELS_FILE=CIFAR-10_standard_test_labels.bin

python3 measurements/run_config.py measurements/configs/artifacts/truncation/figure16 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

fi

if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Table5" ]; then

echo "===Starting 3PC measurements for table 5==="

export MODEL_FILE=Cifar_adam_001/VGG16_CIFAR-10_standard_best.bin
export SAMPLES_FILE=CIFAR-10_standard_test_images.bin
export LABELS_FILE=CIFAR-10_standard_test_labels.bin



python3 measurements/run_config.py measurements/configs/artifacts/truncation/table5 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

fi

if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Table6" ]; then

echo "===Starting 3PC measurements for table 6==="

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table6 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

fi

if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "Table8" ]; then

echo "===Starting 3PC measurements for table 8 with 1Gbps bandwidth and 2ms latency==="

measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -p $PID -l 2 -B 1000 -L 2

python3 measurements/run_config.py measurements/configs/artifacts/truncation/table8 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 $REDUCED

measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -p $PID -L -1 -B -1 #Reset network shaping

fi

echo "=====Finished all 3PC measurements====="
