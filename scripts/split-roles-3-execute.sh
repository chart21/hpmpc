#!/bin/bash

helpFunction()
{
   echo "Script to run 6 mixed constellations of a 3-PC protocol in parallel"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-g Number of GPUs to use"
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:g:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP0="$OPTARG" ;;
      b ) IP1="$OPTARG" ;;
      c ) IP2="$OPTARG" ;;
      g ) NUM_GPUS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

O_IP0="127.0.0.1"
O_IP1="127.0.0.1"
O_IP2="127.0.0.1"

if [ ! -z "$IP0" ]; then O_IP0="$IP0"; fi
if [ ! -z "$IP1" ]; then O_IP1="$IP1"; fi
if [ ! -z "$IP2" ]; then O_IP2="$IP2"; fi

O_NUM_GPUS=1
if [ ! -z "$NUM_GPUS" ]; then O_NUM_GPUS="$NUM_GPUS"; fi

if [ $O_NUM_GPUS -gt 1 ]; then
# Run executables with GPU support
if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ]; then
    CUDA_VISIBLE_DEVICES=$((0 % O_NUM_GPUS)) ./executables/run-P0--0-1-2.o $O_IP1 $O_IP2 &
    CUDA_VISIBLE_DEVICES=$((1 % O_NUM_GPUS)) ./executables/run-P0--0-2-1.o $O_IP2 $O_IP1 &
    CUDA_VISIBLE_DEVICES=$((2 % O_NUM_GPUS)) ./executables/run-P0--1-0-2.o $O_IP1 $O_IP2 &
    CUDA_VISIBLE_DEVICES=$((3 % O_NUM_GPUS)) ./executables/run-P0--1-2-0.o $O_IP1 $O_IP2 &
    CUDA_VISIBLE_DEVICES=$((4 % O_NUM_GPUS)) ./executables/run-P0--2-0-1.o $O_IP2 $O_IP1 &
    CUDA_VISIBLE_DEVICES=$((5 % O_NUM_GPUS)) ./executables/run-P0--2-1-0.o $O_IP2 $O_IP1 &
fi
if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ]; then
    CUDA_VISIBLE_DEVICES=$((0 % O_NUM_GPUS)) ./executables/run-P1--0-1-2.o $O_IP0 $O_IP2 &
    CUDA_VISIBLE_DEVICES=$((1 % O_NUM_GPUS)) ./executables/run-P1--0-2-1.o $O_IP0 $O_IP2 &
    CUDA_VISIBLE_DEVICES=$((2 % O_NUM_GPUS)) ./executables/run-P1--1-0-2.o $O_IP0 $O_IP2 &
    CUDA_VISIBLE_DEVICES=$((3 % O_NUM_GPUS)) ./executables/run-P1--1-2-0.o $O_IP2 $O_IP0 &
    CUDA_VISIBLE_DEVICES=$((4 % O_NUM_GPUS)) ./executables/run-P1--2-0-1.o $O_IP2 $O_IP0 &
    CUDA_VISIBLE_DEVICES=$((5 % O_NUM_GPUS)) ./executables/run-P1--2-1-0.o $O_IP2 $O_IP0 &
fi
if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ]; then
    CUDA_VISIBLE_DEVICES=$((0 % O_NUM_GPUS)) ./executables/run-P2--0-1-2.o $O_IP0 $O_IP1 &
    CUDA_VISIBLE_DEVICES=$((1 % O_NUM_GPUS)) ./executables/run-P2--0-2-1.o $O_IP0 $O_IP1 &
    CUDA_VISIBLE_DEVICES=$((2 % O_NUM_GPUS)) ./executables/run-P2--1-0-2.o $O_IP1 $O_IP0 &
    CUDA_VISIBLE_DEVICES=$((3 % O_NUM_GPUS)) ./executables/run-P2--1-2-0.o $O_IP1 $O_IP0 &
    CUDA_VISIBLE_DEVICES=$((4 % O_NUM_GPUS)) ./executables/run-P2--2-0-1.o $O_IP0 $O_IP1 &
    CUDA_VISIBLE_DEVICES=$((5 % O_NUM_GPUS)) ./executables/run-P2--2-1-0.o $O_IP1 $O_IP0 &
fi

# Run executables without GPU support
else
    if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ]; then
        ./executables/run-P0--0-1-2.o $O_IP1 $O_IP2 &
        ./executables/run-P0--0-2-1.o $O_IP2 $O_IP1 &
        ./executables/run-P0--1-0-2.o $O_IP1 $O_IP2 &
        ./executables/run-P0--1-2-0.o $O_IP1 $O_IP2 &
        ./executables/run-P0--2-0-1.o $O_IP2 $O_IP1 &
        ./executables/run-P0--2-1-0.o $O_IP2 $O_IP1 &
    fi
    if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ]; then
        ./executables/run-P1--0-1-2.o $O_IP0 $O_IP2 &
        ./executables/run-P1--0-2-1.o $O_IP0 $O_IP2 &
        ./executables/run-P1--1-0-2.o $O_IP0 $O_IP2 &
        ./executables/run-P1--1-2-0.o $O_IP2 $O_IP0 &
        ./executables/run-P1--2-0-1.o $O_IP2 $O_IP0 &
        ./executables/run-P1--2-1-0.o $O_IP2 $O_IP0 &
    fi
    if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ]; then
        ./executables/run-P2--0-1-2.o $O_IP0 $O_IP1 &
        ./executables/run-P2--0-2-1.o $O_IP0 $O_IP1 &
        ./executables/run-P2--1-0-2.o $O_IP1 $O_IP0 &
        ./executables/run-P2--1-2-0.o $O_IP1 $O_IP0 &
        ./executables/run-P2--2-0-1.o $O_IP0 $O_IP1 &
        ./executables/run-P2--2-1-0.o $O_IP1 $O_IP0 &
    fi
fi


FAIL=0
for job in $(jobs -p); do
    wait "$job" || ((++FAIL))
done

if [ "$FAIL" -eq 0 ]; then
    echo "No errors in Split roles execution"
else
    echo "Errors detected in Split roles execution"
fi

