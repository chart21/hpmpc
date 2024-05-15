#!/bin/bash
helpFunction()
{
   echo "Script to compile and run 24 mixed constellations of a 3-PC protocol with 4 players in parallel"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-d IP address of player 3 (if ip matches player_id can be empty)"
   echo -e "\t-g Number of GPUs to use"     
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:g:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP1="$OPTARG" ;;
      b ) IP2="$OPTARG" ;;
      c ) IP3="$OPTARG" ;;
      d ) IP4="$OPTARG" ;;
      g ) NUM_GPUS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


O_IP1="127.0.0.1"
O_IP2="127.0.0.1"
O_IP3="127.0.0.1"
O_IP4="127.0.0.1"

if [ ! -z "$IP1" ];
then
O_IP1="$IP1"
fi

if [ ! -z "$IP2" ];
then
O_IP2="$IP2"
fi

if [ ! -z "$IP3" ];
then
O_IP3="$IP3"
fi

if [ ! -z "$IP4" ];
then
O_IP4="$IP4"
fi

if [ ! -z "$NUM_GPUS" ];
then
O_NUM_GPUS="$NUM_GPUS"

if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ];
then
CUDA_VISIBLE_DEVICES=$((0 % O_NUM_GPUS)) ./run-P0--1-2-3-4.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((1 % O_NUM_GPUS)) ./run-P0--1-3-2-4.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((2 % O_NUM_GPUS)) ./run-P0--2-1-3-4.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((3 % O_NUM_GPUS)) ./run-P0--2-3-1-4.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((4 % O_NUM_GPUS)) ./run-P0--3-1-2-4.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((5 % O_NUM_GPUS)) ./run-P0--3-2-1-4.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((6 % O_NUM_GPUS)) ./run-P0--1-2-4-3.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((7 % O_NUM_GPUS)) ./run-P0--1-4-2-3.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((8 % O_NUM_GPUS)) ./run-P0--2-1-4-3.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((9 % O_NUM_GPUS)) ./run-P0--2-4-1-3.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((10 % O_NUM_GPUS)) ./run-P0--4-1-2-3.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((11 % O_NUM_GPUS)) ./run-P0--4-2-1-3.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((12 % O_NUM_GPUS)) ./run-P0--1-3-4-2.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((13 % O_NUM_GPUS)) ./run-P0--1-4-3-2.o $O_IP4 $O_IP3 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((14 % O_NUM_GPUS)) ./run-P0--3-1-4-2.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((15 % O_NUM_GPUS)) ./run-P0--3-4-1-2.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((16 % O_NUM_GPUS)) ./run-P0--4-1-3-2.o $O_IP4 $O_IP3 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((17 % O_NUM_GPUS)) ./run-P0--4-3-1-2.o $O_IP4 $O_IP3 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((18 % O_NUM_GPUS)) ./run-P0--2-3-4-1.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((19 % O_NUM_GPUS)) ./run-P0--2-4-3-1.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((20 % O_NUM_GPUS)) ./run-P0--3-2-4-1.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((21 % O_NUM_GPUS)) ./run-P0--3-4-2-1.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((22 % O_NUM_GPUS)) ./run-P0--4-2-3-1.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((23 % O_NUM_GPUS)) ./run-P0--4-3-2-1.o $O_IP4 $O_IP3 $O_IP2 &
fi
if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ];
then
CUDA_VISIBLE_DEVICES=$((0 % O_NUM_GPUS)) ./run-P1--1-2-3-4.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((1 % O_NUM_GPUS)) ./run-P1--1-3-2-4.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((2 % O_NUM_GPUS)) ./run-P1--2-1-3-4.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((3 % O_NUM_GPUS)) ./run-P1--2-3-1-4.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((4 % O_NUM_GPUS)) ./run-P1--3-1-2-4.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((5 % O_NUM_GPUS)) ./run-P1--3-2-1-4.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((6 % O_NUM_GPUS)) ./run-P1--1-2-4-3.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((7 % O_NUM_GPUS)) ./run-P1--1-4-2-3.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((8 % O_NUM_GPUS)) ./run-P1--2-1-4-3.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((9 % O_NUM_GPUS)) ./run-P1--2-4-1-3.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((10 % O_NUM_GPUS)) ./run-P1--4-1-2-3.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((11 % O_NUM_GPUS)) ./run-P1--4-2-1-3.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((12 % O_NUM_GPUS)) ./run-P1--1-3-4-2.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((13 % O_NUM_GPUS)) ./run-P1--1-4-3-2.o $O_IP4 $O_IP3 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((14 % O_NUM_GPUS)) ./run-P1--3-1-4-2.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((15 % O_NUM_GPUS)) ./run-P1--3-4-1-2.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((16 % O_NUM_GPUS)) ./run-P1--4-1-3-2.o $O_IP4 $O_IP3 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((17 % O_NUM_GPUS)) ./run-P1--4-3-1-2.o $O_IP4 $O_IP3 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((18 % O_NUM_GPUS)) ./run-P1--2-3-4-1.o $O_IP2 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((19 % O_NUM_GPUS)) ./run-P1--2-4-3-1.o $O_IP2 $O_IP4 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((20 % O_NUM_GPUS)) ./run-P1--3-2-4-1.o $O_IP3 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((21 % O_NUM_GPUS)) ./run-P1--3-4-2-1.o $O_IP3 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((22 % O_NUM_GPUS)) ./run-P1--4-2-3-1.o $O_IP4 $O_IP2 $O_IP3 &
CUDA_VISIBLE_DEVICES=$((23 % O_NUM_GPUS)) ./run-P1--4-3-2-1.o $O_IP4 $O_IP3 $O_IP2 &
fi
if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ];
then
CUDA_VISIBLE_DEVICES=$((0 % O_NUM_GPUS)) ./run-P2--1-2-3-4.o $O_IP1 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((1 % O_NUM_GPUS)) ./run-P2--1-3-2-4.o $O_IP1 $O_IP3 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((2 % O_NUM_GPUS)) ./run-P2--2-1-3-4.o $O_IP2 $O_IP1 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((3 % O_NUM_GPUS)) ./run-P2--2-3-1-4.o $O_IP2 $O_IP1 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((4 % O_NUM_GPUS)) ./run-P2--3-1-2-4.o $O_IP1 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((5 % O_NUM_GPUS)) ./run-P2--3-2-1-4.o $O_IP2 $O_IP1 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((6 % O_NUM_GPUS)) ./run-P2--1-2-4-3.o $O_IP1 $O_IP2 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((7 % O_NUM_GPUS)) ./run-P2--1-4-2-3.o $O_IP1 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((8 % O_NUM_GPUS)) ./run-P2--2-1-4-3.o $O_IP2 $O_IP1 $O_IP4 &
CUDA_VISIBLE_DEVICES=$((9 % O_NUM_GPUS)) ./run-P2--2-4-1-3.o $O_IP2 $O_IP4 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((10 % O_NUM_GPUS)) ./run-P2--4-1-2-3.o $O_IP4 $O_IP1 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((11 % O_NUM_GPUS)) ./run-P2--4-2-1-3.o $O_IP4 $O_IP2 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((12 % O_NUM_GPUS)) ./run-P2--1-3-4-2.o $O_IP1 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((13 % O_NUM_GPUS)) ./run-P2--1-4-3-2.o $O_IP1 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((14 % O_NUM_GPUS)) ./run-P2--3-1-4-2.o $O_IP1 $O_IP4 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((15 % O_NUM_GPUS)) ./run-P2--3-4-1-2.o $O_IP3 $O_IP1 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((16 % O_NUM_GPUS)) ./run-P2--4-1-3-2.o $O_IP4 $O_IP1 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((17 % O_NUM_GPUS)) ./run-P2--4-3-1-2.o $O_IP4 $O_IP1 $O_IP2 &
CUDA_VISIBLE_DEVICES=$((18 % O_NUM_GPUS)) ./run-P2--2-3-4-1.o $O_IP2 $O_IP3 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((19 % O_NUM_GPUS)) ./run-P2--2-4-3-1.o $O_IP2 $O_IP4 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((20 % O_NUM_GPUS)) ./run-P2--3-2-4-1.o $O_IP3 $O_IP2 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((21 % O_NUM_GPUS)) ./run-P2--3-4-2-1.o $O_IP3 $O_IP4 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((22 % O_NUM_GPUS)) ./run-P2--4-2-3-1.o $O_IP4 $O_IP2 $O_IP1 &
CUDA_VISIBLE_DEVICES=$((23 % O_NUM_GPUS)) ./run-P2--4-3-2-1.o $O_IP4 $O_IP3 $O_IP1 &
fi


else

# Run all executables for P1
if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P1--1-2-3-4.o $O_IP2 $O_IP3 $O_IP4 & 
    ./run-P1--1-3-2-4.o $O_IP3 $O_IP2 $O_IP4 &
    ./run-P1--2-1-3-4.o $O_IP2 $O_IP3 $O_IP4 &
    ./run-P1--2-3-1-4.o $O_IP2 $O_IP3 $O_IP4 &
    ./run-P1--3-1-2-4.o $O_IP3 $O_IP2 $O_IP4 &
    ./run-P1--3-2-1-4.o $O_IP3 $O_IP2 $O_IP4 &
    ./run-P1--1-2-4-3.o $O_IP2 $O_IP4 $O_IP3 &
    ./run-P1--1-4-2-3.o $O_IP4 $O_IP2 $O_IP3 &
    ./run-P1--2-1-4-3.o $O_IP2 $O_IP4 $O_IP3 &
    ./run-P1--2-4-1-3.o $O_IP2 $O_IP4 $O_IP3 &
    ./run-P1--4-1-2-3.o $O_IP4 $O_IP2 $O_IP3 &
    ./run-P1--4-2-1-3.o $O_IP4 $O_IP2 $O_IP3 &
    ./run-P1--1-3-4-2.o $O_IP3 $O_IP4 $O_IP2 &
    ./run-P1--1-4-3-2.o $O_IP4 $O_IP3 $O_IP2 &
    ./run-P1--3-1-4-2.o $O_IP3 $O_IP4 $O_IP2 &
    ./run-P1--3-4-1-2.o $O_IP3 $O_IP4 $O_IP2 &
    ./run-P1--4-1-3-2.o $O_IP4 $O_IP3 $O_IP2 &
    ./run-P1--4-3-1-2.o $O_IP4 $O_IP3 $O_IP2 &
    ./run-P1--2-3-4-1.o $O_IP2 $O_IP3 $O_IP4 &
    ./run-P1--2-4-3-1.o $O_IP2 $O_IP4 $O_IP3 &
    ./run-P1--3-2-4-1.o $O_IP3 $O_IP2 $O_IP4 &
    ./run-P1--3-4-2-1.o $O_IP3 $O_IP4 $O_IP2 &
    ./run-P1--4-2-3-1.o $O_IP4 $O_IP2 $O_IP3 &
    ./run-P1--4-3-2-1.o $O_IP4 $O_IP3 $O_IP2 &
fi
# Run all executables for P2
if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P2--1-2-3-4.o $O_IP1 $O_IP3 $O_IP4 &
    ./run-P2--1-3-2-4.o $O_IP1 $O_IP3 $O_IP4 &
    ./run-P2--2-1-3-4.o $O_IP1 $O_IP3 $O_IP4 &
    ./run-P2--2-3-1-4.o $O_IP3 $O_IP1 $O_IP4 &
    ./run-P2--3-1-2-4.o $O_IP3 $O_IP1 $O_IP4 &
    ./run-P2--3-2-1-4.o $O_IP3 $O_IP1 $O_IP4 &
    ./run-P2--1-2-4-3.o $O_IP1 $O_IP4 $O_IP3 &
    ./run-P2--1-4-2-3.o $O_IP1 $O_IP4 $O_IP3 &
    ./run-P2--2-1-4-3.o $O_IP1 $O_IP4 $O_IP3 &
    ./run-P2--2-4-1-3.o $O_IP4 $O_IP1 $O_IP3 &
    ./run-P2--4-1-2-3.o $O_IP4 $O_IP1 $O_IP3 &
    ./run-P2--4-2-1-3.o $O_IP4 $O_IP1 $O_IP3 &
    ./run-P2--1-3-4-2.o $O_IP1 $O_IP3 $O_IP4 &
    ./run-P2--1-4-3-2.o $O_IP1 $O_IP4 $O_IP3 &
    ./run-P2--3-1-4-2.o $O_IP3 $O_IP1 $O_IP4 &
    ./run-P2--3-4-1-2.o $O_IP3 $O_IP4 $O_IP1 &
    ./run-P2--4-1-3-2.o $O_IP4 $O_IP1 $O_IP3 &
    ./run-P2--4-3-1-2.o $O_IP4 $O_IP3 $O_IP1 &
    ./run-P2--2-3-4-1.o $O_IP3 $O_IP4 $O_IP1 &
    ./run-P2--2-4-3-1.o $O_IP4 $O_IP3 $O_IP1 &
    ./run-P2--3-2-4-1.o $O_IP3 $O_IP4 $O_IP1 &
    ./run-P2--3-4-2-1.o $O_IP3 $O_IP4 $O_IP1 &
    ./run-P2--4-2-3-1.o $O_IP4 $O_IP3 $O_IP1 &
    ./run-P2--4-3-2-1.o $O_IP4 $O_IP3 $O_IP1 &
fi
# Run all executables for P3
if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P3--1-2-3-4.o $O_IP1 $O_IP2 $O_IP4 &
    ./run-P3--1-3-2-4.o $O_IP1 $O_IP2 $O_IP4 &
    ./run-P3--2-1-3-4.o $O_IP2 $O_IP1 $O_IP4 &
    ./run-P3--2-3-1-4.o $O_IP2 $O_IP1 $O_IP4 &
    ./run-P3--3-1-2-4.o $O_IP1 $O_IP2 $O_IP4 &
    ./run-P3--3-2-1-4.o $O_IP2 $O_IP1 $O_IP4 &
    ./run-P3--1-2-4-3.o $O_IP1 $O_IP2 $O_IP4 &
    ./run-P3--1-4-2-3.o $O_IP1 $O_IP4 $O_IP2 &
    ./run-P3--2-1-4-3.o $O_IP2 $O_IP1 $O_IP4 &
    ./run-P3--2-4-1-3.o $O_IP2 $O_IP4 $O_IP1 &
    ./run-P3--4-1-2-3.o $O_IP4 $O_IP1 $O_IP2 &
    ./run-P3--4-2-1-3.o $O_IP4 $O_IP2 $O_IP1 &
    ./run-P3--1-3-4-2.o $O_IP1 $O_IP4 $O_IP2 &
    ./run-P3--1-4-3-2.o $O_IP1 $O_IP4 $O_IP2 &
    ./run-P3--3-1-4-2.o $O_IP1 $O_IP4 $O_IP2 &
    ./run-P3--3-4-1-2.o $O_IP4 $O_IP1 $O_IP2 &
    ./run-P3--4-1-3-2.o $O_IP4 $O_IP1 $O_IP2 &
    ./run-P3--4-3-1-2.o $O_IP4 $O_IP1 $O_IP2 &
    ./run-P3--2-3-4-1.o $O_IP2 $O_IP4 $O_IP1 &
    ./run-P3--2-4-3-1.o $O_IP2 $O_IP4 $O_IP1 &
    ./run-P3--3-2-4-1.o $O_IP2 $O_IP4 $O_IP1 &
    ./run-P3--3-4-2-1.o $O_IP4 $O_IP2 $O_IP1 &
    ./run-P3--4-2-3-1.o $O_IP4 $O_IP2 $O_IP1 &
    ./run-P3--4-3-2-1.o $O_IP4 $O_IP2 $O_IP1 &
fi
# Run all executables for P4
if [ "$O_PARTY" = "3" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P4--1-2-3-4.o $O_IP1 $O_IP2 $O_IP3 &
    ./run-P4--1-3-2-4.o $O_IP1 $O_IP3 $O_IP2 &
    ./run-P4--2-1-3-4.o $O_IP2 $O_IP1 $O_IP3 &
    ./run-P4--2-3-1-4.o $O_IP2 $O_IP3 $O_IP1 &
    ./run-P4--3-1-2-4.o $O_IP3 $O_IP1 $O_IP2 &
    ./run-P4--3-2-1-4.o $O_IP3 $O_IP2 $O_IP1 &
    ./run-P4--1-2-4-3.o $O_IP1 $O_IP2 $O_IP3 &
    ./run-P4--1-4-2-3.o $O_IP1 $O_IP2 $O_IP3 &
    ./run-P4--2-1-4-3.o $O_IP2 $O_IP1 $O_IP3 &
    ./run-P4--2-4-1-3.o $O_IP2 $O_IP1 $O_IP3 &
    ./run-P4--4-1-2-3.o $O_IP1 $O_IP2 $O_IP3 &
    ./run-P4--4-2-1-3.o $O_IP2 $O_IP1 $O_IP3 &
    ./run-P4--1-3-4-2.o $O_IP1 $O_IP3 $O_IP2 &
    ./run-P4--1-4-3-2.o $O_IP1 $O_IP3 $O_IP2 &
    ./run-P4--3-1-4-2.o $O_IP3 $O_IP1 $O_IP2 &
    ./run-P4--3-4-1-2.o $O_IP3 $O_IP1 $O_IP2 &
    ./run-P4--4-1-3-2.o $O_IP1 $O_IP3 $O_IP2 &
    ./run-P4--4-3-1-2.o $O_IP3 $O_IP1 $O_IP2 &
    ./run-P4--2-3-4-1.o $O_IP2 $O_IP3 $O_IP1 &
    ./run-P4--2-4-3-1.o $O_IP2 $O_IP3 $O_IP1 &
    ./run-P4--3-2-4-1.o $O_IP3 $O_IP2 $O_IP1 &
    ./run-P4--3-4-2-1.o $O_IP3 $O_IP2 $O_IP1 &
    ./run-P4--4-2-3-1.o $O_IP2 $O_IP3 $O_IP1 &
    ./run-P4--4-3-2-1.o $O_IP3 $O_IP2 $O_IP1 &
    fi

fi

FAIL=0
for job in $(jobs -p); do
# echo $job
    wait "$job" || ((++FAIL))
done


if [ "$FAIL" -eq 0 ];
then
echo "No errors in Split roles ececution"
else
echo "Erros detected in Split roles ececution"
fi

