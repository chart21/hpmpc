#!/bin/bash

##---Adjust these---
protocols=(5) # 5: 3PC
#protocols=(12) # 12: 4PC
use_nvcc=(0 1 2) # 0: CPU-only, 1: GPU for matmul, 2: GPU for Convolution
Dattype=256 # Careful, requires AVX512 support by your CPU architecture. In not supported use 256 (AVX2), 128 (SSE), or 32 (None) for vectorization
num_processes_4PC=1 # Careful, multiplies by 24
pre=(0 1) # 0: no pre-processing, 1: pre-processing
num_repititions=10
##---End of adjust---


num_inputsa=(32 64 128 256 512) # Careful, multiplies with split_role_factor*num_processes*dattypes/bitlength
num_inputsb=(1) # Careful, multiplies with split_role_factor*num_processes*dattypes/bitlength
num_inputsc=(1 2 4 8 16 32 64) # Careful, multiplies with split_role_factor*num_processes*dattypes/bitlength

functiona=400 # do not change
functionb=401 # do not change
functionc=402 # do not change


num_processes_3PC=4*$num_processes_4PC #do not change
split_role_factor_3PC=6 #do not change -> multiplies with num_processes_3PC
split_role_factor_4PC=24 #do not change -> multiplies with num_processes_4PC
bitlength=32 #if you change this, also changed reduced bitlength in config.h






helpFunction()
{
   echo "Script to run neural network inference in batches"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-d IP address of player 3 (if ip matches player_id can be empty)"
   echo -e "\t-x Compiler (g++/clang++/..)"
   echo -e "\t"-u "Comile with nvcc and cutlass GEMM (0/1)"


   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:x:u:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP0="$OPTARG" ;;
      b ) IP1="$OPTARG" ;;
      c ) IP2="$OPTARG" ;;
      d ) IP3="$OPTARG" ;;
      x ) COMPILER="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

O_IP0="127.0.0.1"
O_IP1="127.0.0.1"
O_IP2="127.0.0.1"
O_IP3="127.0.0.1"

if [ ! -z "$IP0" ];
then
O_IP0="$IP0"
fi

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




cp scripts/benchmark/base_config.h config.h 
sed -i -e "s/\(define DATTYPE \).*/\1$Dattype/" config.h
for pr in "${protocols[@]}"
do
    sed -i -e "s/\(define PROTOCOL \).*/\1$pr/" config.h
        sed -i -e "s/\(define FUNCTION_IDENTIFIER \).*/\1$functiona/" config.h
        for num_inputs in "${num_inputsa[@]}"
        do
        batch_size_4PC=$((num_inputs*split_role_factor_4PC*num_processes_4PC*Dattype/bitlength))
        batch_size_3PC=$((num_inputs*split_role_factor_3PC*num_processes_3PC*Dattype/bitlength))
        sed -i -e "s/\(define NUM_INPUTS \).*/\1$num_inputs/" config.h
        for use_nv in "${use_nvcc[@]}"
        do
            sed -i -e "s/\(define USE_CUDA_GEMM \).*/\1$use_nv/" config.h
            if [ "$pr" -gt "6" ]
            then

                    sed -i -e "s/\(define PROCESS_NUM \).*/\1$num_processes_4PC/" config.h
                    ./scripts/split-roles-4-compile.sh -u $use_nv -p $O_PARTY 
            else
                sed -i -e "s/\(define PROCESS_NUM \).*/\1$num_processes_3PC/" config.h
                ./scripts/split-roles-3-compile.sh -u $use_nv -p $O_PARTY 
            fi
    if [ "$use_nv" != "0" ]
    then
        ./scripts/cuda_compile.sh 
    fi
for i in $(seq 1 $num_repititions)
do
    if [ "$pr" -gt "6" ]
    then
        echo "Running protocol $pr, function $functiona, use_nvcc $use_nv, batch_size $batch_size_4PC"
        ./scripts/split-roles-4-execute.sh -p $O_PARTY -a $O_IP0 -b $O_IP1 -c $O_IP2 -d $O_IP3
    else
    echo "Running protocol $pr, function $functiona, use_nvcc $use_nv,  batch_size $batch_size_3PC"
    ./scripts/split-roles-3-execute.sh -p $O_PARTY -a $O_IP0 -b $O_IP1 -c $O_IP2
    fi
    sleep 1
done
done
done
done

for pr in "${protocols[@]}"
do
    sed -i -e "s/\(define PROTOCOL \).*/\1$pr/" config.h
        sed -i -e "s/\(define FUNCTION_IDENTIFIER \).*/\1$functionb/" config.h
        for num_inputs in "${num_inputsb[@]}"
        do
        batch_size_4PC=$((num_inputs*split_role_factor_4PC*num_processes_4PC*Dattype/bitlength))
        batch_size_3PC=$((num_inputs*split_role_factor_3PC*num_processes_3PC*Dattype/bitlength))
        sed -i -e "s/\(define NUM_INPUTS \).*/\1$num_inputs/" config.h
        for use_nv in "${use_nvcc[@]}"
        do
            sed -i -e "s/\(define USE_CUDA_GEMM \).*/\1$use_nv/" config.h
            if [ "$pr" -gt "6" ]
            then

                    sed -i -e "s/\(define PROCESS_NUM \).*/\1$num_processes_4PC/" config.h
                    ./scripts/split-roles-4-compile.sh -u $use_nv -p $O_PARTY 
            else
                sed -i -e "s/\(define PROCESS_NUM \).*/\1$num_processes_3PC/" config.h
                ./scripts/split-roles-3-compile.sh -u $use_nv -p $O_PARTY 
            fi
    if [ "$use_nv" != "0" ]
    then
        ./scripts/cuda_compile.sh 
    fi
    for i in $(seq 1 $num_repititions)
    do
    if [ "$pr" -gt "6" ]
    then
        echo "Running protocol $pr, function $functionb, use_nvcc $use_nv,  batch_size $batch_size_4PC"
        ./scripts/split-roles-4-execute.sh -p $O_PARTY -a $O_IP0 -b $O_IP1 -c $O_IP2 -d $O_IP3
    else
    echo "Running protocol $pr, function $functionb, use_nvcc $use_nv,  batch_size $batch_size_3PC"
    ./scripts/split-roles-3-execute.sh -p $O_PARTY -a $O_IP0 -b $O_IP1 -c $O_IP2
    fi
    sleep 1
done
done
done
done

for pr in "${protocols[@]}"
do
    sed -i -e "s/\(define PROTOCOL \).*/\1$pr/" config.h
        sed -i -e "s/\(define FUNCTION_IDENTIFIER \).*/\1$functionc/" config.h
        for num_inputs in "${num_inputsc[@]}"
        do
        sed -i -e "s/\(define NUM_INPUTS \).*/\1$num_inputs/" config.h
        batch_size_4PC=$((num_inputs*split_role_factor_4PC*num_processes_4PC*Dattype/bitlength))
        batch_size_3PC=$((num_inputs*split_role_factor_3PC*num_processes_3PC*Dattype/bitlength))
        for use_nv in "${use_nvcc[@]}"
        do
                    sed -i -e "s/\(define COMPRESS \).*/\1$rb/" config.h
            if [ "$pr" -gt "6" ]
            then

                    sed -i -e "s/\(define PROCESS_NUM \).*/\1$num_processes_4PC/" config.h
                    ./scripts/split-roles-4-compile.sh -u $use_nv -p $O_PARTY 
            else
                sed -i -e "s/\(define PROCESS_NUM \).*/\1$num_processes_3PC/" config.h
                ./scripts/split-roles-3-compile.sh -u $use_nv -p $O_PARTY 
            fi
    if [ "$use_nv" != "0" ]
    then
        ./scripts/cuda_compile.sh 
    fi
    for i in $(seq 1 $num_repititions)
    do
    if [ "$pr" -gt "6" ]
    then
        echo "Running protocol $pr, function $functionc, use_nvcc $use_nv,  batch_size $batch_size_4PC"
        ./scripts/split-roles-4-execute.sh -p $O_PARTY -a $O_IP0 -b $O_IP1 -c $O_IP2 -d $O_IP3
    else
    echo "Running protocol $pr, function $functionc, use_nvcc $use_nv,  batch_size $batch_size_3PC"
    ./scripts/split-roles-3-execute.sh -p $O_PARTY -a $O_IP0 -b $O_IP1 -c $O_IP2
    fi
    sleep 1
done
done
done
done

cp scripts/benchmark/base_config.h config.h

