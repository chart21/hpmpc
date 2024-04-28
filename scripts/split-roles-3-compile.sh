#!/bin/bash
helpFunction()
{
   echo "Script to compile and run 6 mixed constellations of a 3-PC protocol in parallel"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-x Compiler (g++/clang++/..)"
   echo -e "\t"-u "Comile with nvcc and cutlass GEMM (0/1)"


   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:x:u:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP0="$OPTARG" ;;
      b ) IP1="$OPTARG" ;;
      c ) IP2="$OPTARG" ;;
      x ) COMPILER="$OPTARG" ;;
      u ) USE_NVCC="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

comp="g++"
if [ ! -z "$COMPILER" ]
then
comp="$COMPILER"
fi

flags="-w -march=native -Ofast -fno-finite-math-only -std=c++2a -pthread -I SimpleNN -lstdc++fs"
# flags="-march=native -Ofast -std=c++2a -pthread -lssl -lcrypto"
if [ "$USE_NVCC" -gt "0" ]
then
    flags=$flags" -c"
fi

flags="$flags MP-SPDZ/lib/help/Util.cpp -DNDEBUG"

O_IP0="127.0.0.1"
O_IP1="127.0.0.1"
O_IP2="127.0.0.1"

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


# Compile all executables for P0
if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P0 ..."
    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P0--0-1-2.o $flags
    
    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P0--0-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P0--1-0-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P0--1-2-0.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P0--2-0-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P0--2-1-0.o $flags

fi

# Compile all executables for -3P2
if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P1 ..."
    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P1--0-1-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P1--0-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P1--1-0-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P1--1-2-0.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P1--2-0-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P1--2-1-0.o $flags

fi

# Compile all executables for P2
if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P2 ..."
    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P2--0-1-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P2--0-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P2--1-0-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P2--1-2-0.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P2--2-0-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P2--2-1-0.o $flags

fi

sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h

echo "Finished compiling"
