#!/bin/bash
helpFunction()
{
   echo "Script to configure and compile executables for a run."
   echo -e "Only arguments you want to change have to be set."
   echo -e "\t-n Number of elements"
   echo -e "\t-a Default Bitlength of integers"
   echo -e "\t-b base_port: Needs to be the same for all players for successful networking (e.g. 6000)"
   echo -e "\t-d Datatype used for slicing: 1(bool),8(char),64(uint64),128(SSE),256(AVX),512(AVX512)"
   echo -e "\t-p Player ID (0/1/2). Use all3 or all4 for compilling all players"
   echo -e "\t-f Function Idenftifier (0: run, 2: AND, ...)"
   echo -e "\t-c Pack Bool in Char before sending? (0/1). Only used with -d 1"
   echo -e "\t-s MPC Protocol (1(Sharemind),2(Replicated),3(Astra),4(OEC DUP),5(OEC REP),6(TTP))"
   echo -e "\t-i Initialize circuit seperatly (0) or at runtime (1)?"
   echo -e "\t-l Include the Online Phase in this executable  (0/1)?"
   echo -e "\t-e Compile circuit with Preprocessing phase before online phase  (0/1)?"
   echo -e "\t-o Use additional assumptions to optimize the sharing phase? (0/1)"
   echo -e "\t-g Compile flags (other than standard)"
   echo -e "\t-x Compiler (g++/clang++/..)"
   echo -e "\t-h USE SSL? (0/1)"
   echo -e "\t-j Number of parallel processes to use"
   echo -e "\t-v Random Number Generator (0: XOR_Shift/1 AES Bitsliced/2: AES_NI)"
   echo -e "\t-t Total Timeout in seconds for attempting to connect to a player"
   echo -e "\t-m VERIFY_BUFFER: How many gates should be buffered until verifying them? 0 means the data of an entire communication round is buffered "
   echo -e "\t-k Timeout in millisecond before attempting to connect again to a socket "
   echo -e "\t-y SEND_BUFFER: How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire communication round is buffered
"
   echo -e "\t-z RECV_BUFFER: How many reciving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
"
echo -e "\t"-u "Comile with nvcc and cutlass GEMM (0/1)"
   exit 1 # Exit script after printing help
}

while getopts "b:a:d:c:f:n:s:i:l:p:o:g:x:e:h:j:v:t:m:k:y:z:u:" opt
do
   case "$opt" in
      b ) BASE_PORT="$OPTARG" ;;
      a ) BITLENGTH="$OPTARG" ;;
      d ) DATTYPE="$OPTARG" ;;
      c ) COMPRESS="$OPTARG" ;;
      f ) FUNCTION_IDENTIFIER="$OPTARG" ;;
      n ) NUM_INPUTS="$OPTARG" ;;
      s ) PROTOCOL="$OPTARG" ;;
      i ) INIT="$OPTARG" ;;
      l ) LIVE="$OPTARG" ;;
      p ) PARTY="$OPTARG" ;;
      o ) OPT_SHARE="$OPTARG" ;;
      g ) GNU_OPTIONS="$OPTARG" ;;
      x ) COMPILER="$OPTARG" ;;
      e ) PREPROCESSING="$OPTARG" ;;
      h ) USE_SSL="$OPTARG" ;;
      j ) PROCESS_NUM="$OPTARG" ;;
      v ) RANDOM_ALGORITHM="$OPTARG" ;;
      t ) CONNECTION_TIMEOUT="$OPTARG" ;;
      m ) VERIFY_BUFFER="$OPTARG" ;;
      k ) CONNECTION_RETRY="$OPTARG" ;;
      y ) SEND_BUFFER="$OPTARG" ;;
      z ) RECV_BUFFER="$OPTARG" ;;
      u ) USE_NVCC="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

comp="g++"
if [ ! -z "$COMPILER" ]
then
comp="$COMPILER"
fi

ssl="0"
if [ ! -z "$USE_SSL" ]
then
ssl="$USE_SSL"
fi

use_nvcc="0"
if [ ! -z "$USE_NVCC" ]
then
use_nvcc="$USE_NVCC"
fi

if [ "$ssl" = "1" ]
then
    flags="-w -march=native -Ofast -fno-finite-math-only -std=c++20 -pthread -lssl -lcrypto -I SimpleNN" # -lstdc++fs might be needed on some systems
else
    flags="-w -march=native -Ofast -fno-finite-math-only -std=c++20 -pthread -I SimpleNN" # -lstdc++fs might be needed on some systems
fi

if [ "$use_nvcc" -gt "0" ]
then
    flags=$flags" -c"
fi

if [ ! -z "$GNU_OPTIONS" ]
then
    flags=$flags" "$GNU_OPTIONS
fi

flags="$flags MP-SPDZ/lib/help/Util.cpp -DNDEBUG"

# Begin script in case all parameters are correct

if [ ! -z "$BASE_PORT" ]
then
    sed -i -e "s/\(define BASE_PORT \).*/\1$BASE_PORT/" config.h
fi
if [ ! -z "$BITLENGTH" ]
then
    sed -i -e "s/\(define BITLENGTH \).*/\1$BITLENGTH/" config.h
fi
if [ ! -z "$NUM_INPUTS" ]
then
    sed -i -e "s/\(define NUM_INPUTS \).*/\1$NUM_INPUTS/" config.h
fi
if [ ! -z "$DATTYPE" ]
then
    sed -i -e "s/\(define DATTYPE \).*/\1$DATTYPE/" config.h
fi
if [ ! -z "$FUNCTION_IDENTIFIER" ]
then
    sed -i -e "s/\(define FUNCTION_IDENTIFIER \).*/\1$FUNCTION_IDENTIFIER/" config.h
fi
if [ ! -z "$COMPRESS" ]
then
    sed -i -e "s/\(define COMPRESS \).*/\1$COMPRESS/" config.h
fi
if [ ! -z "$LIVE" ]
then
    sed -i -e "s/\(define LIVE \).*/\1$LIVE/" config.h
fi
if [ ! -z "$INIT" ]
then
    sed -i -e "s/\(define INIT \).*/\1$INIT/" config.h
fi
if [ ! -z "$OPT_SHARE" ]
then
    sed -i -e "s/\(define OPT_SHARE \).*/\1$OPT_SHARE/" config.h
fi
if [ ! -z "$PROTOCOL" ]
then
    sed -i -e "s/\(define PROTOCOL \).*/\1$PROTOCOL/" config.h
fi
if [ ! -z "$PARTY" ]
then
    sed -i -e "s/\(define PARTY \).*/\1$PARTY/" config.h
fi
if [ ! -z "$NUM_PLAYERS" ]
then
    sed -i -e "s/\(define num_players \).*/\1$NUM_PLAYERS/" config.h
fi
if [ ! -z "$PREPROCESSING" ]
then
    sed -i -e "s/\(define PRE \).*/\1$PREPROCESSING/" config.h
fi
if [ ! -z "$RANDOM_ALGORITHM" ]
then
    sed -i -e "s/\(define RANDOM_ALGORITHM \).*/\1$RANDOM_ALGORITHM/" config.h
fi

if [ ! -z "$USE_SSL" ]
then
    sed -i -e "s/\(define USE_SSL \).*/\1$USE_SSL/" config.h
fi

if [ ! -z "$PROCESS_NUM" ]
then
    sed -i -e "s/\(define PROCESS_NUM \).*/\1$PROCESS_NUM/" config.h
fi
if [ ! -z "$CONNECTION_TIMEOUT" ]
then
    sed -i -e "s/\(define CONNECTION_TIMEOUT \).*/\1$CONNECTION_TIMEOUT/" config.h
fi

if [ ! -z "$CONNECTION_RETRY" ]
then
    sed -i -e "s/\(define CONNECTION_RETRY \).*/\1$CONNECTION_RETRY/" config.h
fi

if [ ! -z "$SEND_BUFFER" ]
then
    sed -i -e "s/\(define SEND_BUFFER \).*/\1$SEND_BUFFER/" config.h
fi

if [ ! -z "$RECV_BUFFER" ]
then
    sed -i -e "s/\(define RECV_BUFFER \).*/\1$RECV_BUFFER/" config.h
fi

if [ ! -z "$VERIFY_BUFFER" ]
then
    sed -i -e "s/\(define VERIFY_BUFFER \).*/\1$VERIFY_BUFFER/" config.h
fi

for i in {0..2}
do
    if [ "$i" = "$PARTY" ] || [ "$PARTY" = "all3" ] || [ "$PARTY" = "all4" ];
    then
        if [ "$PARTY" = "all3" ] || [ "$PARTY" = "all4" ];
        then
            sed -i -e "s/\(define PARTY \).*/\1"$i"/" config.h
        fi
        if [ "$LIVE" = "0" ] && [ "$INIT" = "1" ];
        then
            sed -i -e "s/\(LIVE \).*/\10/" config.h
            sed -i -e "s/\(INIT \).*/\11/" config.h
            echo "Compiling INIT executable for P-$i ..."
            "$comp" main.cpp -o ./executables/run-P"$i"-INIT.o $flags
            ./executables/run-P"$i"-INIT.o
            sed -i -e "s/\(LIVE \).*/\11/" config.h
            sed -i -e "s/\(INIT \).*/\10/" config.h
        fi
            echo "Compiling executable for P-$i ..."
            "$comp" main.cpp -o ./executables/run-P"$i".o $flags
    fi
done

if [ "$PARTY" = "all4" ] || [ "$PARTY" = "3" ];
then
        sed -i -e "s/\(define PARTY \).*/\1"3"/" config.h
        if [ "$LIVE" = "0" ] && [ "$INIT" = "1" ];
        then
            sed -i -e "s/\(LIVE \).*/\10/" config.h
            sed -i -e "s/\(INIT \).*/\11/" config.h
            echo "Compiling INIT executable for P-3 ..."
            "$comp" main.cpp -o ./executables/run-P3-INIT.o $flags
            ./executables/run-P"$i"-INIT.o
            sed -i -e "s/\(LIVE \).*/\11/" config.h
            sed -i -e "s/\(INIT \).*/\10/" config.h
        fi
            echo "Compiling executable for P-3 ..."
            "$comp" main.cpp -o ./executables/run-P3.o $flags
    fi

echo "Finished compiling"
