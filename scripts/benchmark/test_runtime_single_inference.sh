#!/bin/bash

##---Adjust these---
# protocols=(5 12) # 5: 3PC, 12: 4PC
protocols=(5)
functions=(70 170 270 71 171 271 72 172 272 73 173 273 74 174 274 75 175 275 76 176 276 77 177 277 78 178 278 79 179 279 80 180 280 81 181 281 82 182 282 83 183 283)
# functions=(82)
use_nvcc=(0 1 2) # 0: CPU-only, 1: GPU for matmul, 2: GPU for Convolution
reduced_bitlength=(0 1) # 0: full bitlength, 1: reduced bitlength (8-bit by default)
pre=(0 1) # 0: no pre-processing, 1: pre-processing
##---End of adjust---


helpFunction()
{
   echo "Script to test the runtime of single inferences for different configurations"
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

for pr in "${protocols[@]}"
do
    sed -i -e "s/\(define PROTOCOL \).*/\1$pr/" config.h
    for f in "${functions[@]}"
    do
        sed -i -e "s/\(define FUNCTION_IDENTIFIER \).*/\1$f/" config.h
        for use_nv in "${use_nvcc[@]}"
        do
            sed -i -e "s/\(define USE_CUDA_GEMM \).*/\1$use_nv/" config.h
            for rb in "${reduced_bitlength[@]}"
            do
            for prep in "${pre[@]}"
            do
            sed -i -e "s/\(define PRE \).*/\1$prep/" config.h
            if [[ "$O_PARTY" == *"all"* ]]
            then 
                if [ "$pr" -gt "6" ]
                then
                    ./scripts/config.sh -f $f -p all4 -u $use_nv -c $rb
                else
                    ./scripts/config.sh -f $f -p all3 -u $use_nv -c $rb
                fi
            else
                ./scripts/config.sh -f $f -p $O_PARTY -u $use_nv -c $rb
            fi
            
            if [ "$use_nv" != "0" ]
            then
                ./scripts/cuda_compile.sh
            fi

            echo "Running protocol $pr, function $f, use_nvcc $use_nv, reduced_bitlength $rb, pre $prep"
            
                #if pr > 6
            if [ "$pr" -gt "6" ]
            then
                if [ "$O_PARTY" == "0" ]
                then
                    ./run-P0.o $O_IP1 $O_IP2 $O_IP3
                elif [ "$O_PARTY" == "1" ]
                then
                    ./run-P1.o $O_IP0 $O_IP2 $O_IP3
                elif [ "$O_PARTY" == "2" ]
                then
                    ./run-P2.o $O_IP0 $O_IP1 $O_IP3
                elif [ "$O_PARTY" == "3" ]
                then
                    ./run-P3.o $O_IP0 $O_IP1 $O_IP2
                elif [ "$O_PARTY" == *"all"* ]
                then
                    ./run-P0.o & 
                    ./run-P1.o &
                    ./run-P2.o &
                    ./run-P3.o &
                    wait
                else
                    echo "Invalid party number"
                fi
            else 
                if [ "$O_PARTY" == "0" ]
                then
                    ./run-P0.o $O_IP1 $O_IP2
                elif [ "$O_PARTY" == "1" ]
                then
                    ./run-P1.o $O_IP0 $O_IP2
                elif [ "$O_PARTY" == "2" ]
                then
                    ./run-P2.o $O_IP0 $O_IP1
                elif [ "$O_PARTY" == "3" ]
                then
                    ./run-P3.o $O_IP0 $O_IP1
                elif [[ "$O_PARTY" == *"all"* ]]
                then
                    ./run-P0.o &
                    ./run-P1.o &
                    ./run-P2.o &
                    wait
                else
                    echo "Invalid party number"
                fi
            fi
        done
        done
    done
done
done
cp scripts/benchmark/base_config.h config.h 

