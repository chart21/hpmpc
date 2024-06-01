#!/bin/bash
helpFunction()
{
   echo "Script to compile and run 24 mixed constellations of a 3-PC protocol with 4 players in parallel"
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
      a ) IP1="$OPTARG" ;;
      b ) IP2="$OPTARG" ;;
      c ) IP3="$OPTARG" ;;
      d ) IP4="$OPTARG" ;;
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

flags="-march=native -Ofast -fno-finite-math-only -std=c++2a -pthread -I SimpleNN -lstdc++fs"
# flags="-march=native -Ofast -std=c++2a -pthread -lssl -lcrypto"
if [ "$USE_NVCC" -gt "0" ]
then
    flags=$flags" -c"
fi

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





# Compile all executables for P0
if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P0 ..."
    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P1--1-2-3-4.o $flags
    
    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P1--1-3-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P1--2-1-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P1--2-3-1-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P1--3-1-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P1--3-2-1-4.o $flags

    

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"6"/" config.h
    "$comp" main.cpp -o ./run-P1--1-2-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"7"/" config.h
    "$comp" main.cpp -o ./run-P1--1-4-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"8"/" config.h
    "$comp" main.cpp -o ./run-P1--2-1-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"9"/" config.h
    "$comp" main.cpp -o ./run-P1--2-4-1-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"10"/" config.h
    "$comp" main.cpp -o ./run-P1--4-1-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"11"/" config.h
    "$comp" main.cpp -o ./run-P1--4-2-1-3.o $flags


    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"12"/" config.h
    "$comp" main.cpp -o ./run-P1--1-3-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"13"/" config.h
    "$comp" main.cpp -o ./run-P1--1-4-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"14"/" config.h
    "$comp" main.cpp -o ./run-P1--3-1-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"15"/" config.h
    "$comp" main.cpp -o ./run-P1--3-4-1-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"16"/" config.h
    "$comp" main.cpp -o ./run-P1--4-1-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"17"/" config.h
    "$comp" main.cpp -o ./run-P1--4-3-1-2.o $flags


        sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"18"/" config.h
    "$comp" main.cpp -o ./run-P1--2-3-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"19"/" config.h
    "$comp" main.cpp -o ./run-P1--2-4-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"20"/" config.h
    "$comp" main.cpp -o ./run-P1--3-2-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"21"/" config.h
    "$comp" main.cpp -o ./run-P1--3-4-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"22"/" config.h
    "$comp" main.cpp -o ./run-P1--4-2-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"23"/" config.h
    "$comp" main.cpp -o ./run-P1--4-3-2-1.o $flags

fi

# Compile all executables for -3P2
if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P1 ..."
    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P2--1-2-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P2--1-3-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P2--2-1-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P2--2-3-1-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P2--3-1-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P2--3-2-1-4.o $flags


    
    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"6"/" config.h
    "$comp" main.cpp -o ./run-P2--1-2-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"7"/" config.h
    "$comp" main.cpp -o ./run-P2--1-4-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"8"/" config.h
    "$comp" main.cpp -o ./run-P2--2-1-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"9"/" config.h
    "$comp" main.cpp -o ./run-P2--2-4-1-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"10"/" config.h
    "$comp" main.cpp -o ./run-P2--4-1-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"11"/" config.h
    "$comp" main.cpp -o ./run-P2--4-2-1-3.o $flags



    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"18"/" config.h
    "$comp" main.cpp -o ./run-P2--2-3-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"19"/" config.h
    "$comp" main.cpp -o ./run-P2--2-4-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"20"/" config.h
    "$comp" main.cpp -o ./run-P2--3-2-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"21"/" config.h
    "$comp" main.cpp -o ./run-P2--3-4-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"22"/" config.h
    "$comp" main.cpp -o ./run-P2--4-2-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"23"/" config.h
    "$comp" main.cpp -o ./run-P2--4-3-2-1.o $flags


    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"12"/" config.h
    "$comp" main.cpp -o ./run-P2--1-3-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"13"/" config.h
    "$comp" main.cpp -o ./run-P2--1-4-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"14"/" config.h
    "$comp" main.cpp -o ./run-P2--3-1-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"15"/" config.h
    "$comp" main.cpp -o ./run-P2--3-4-1-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"16"/" config.h
    "$comp" main.cpp -o ./run-P2--4-1-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"17"/" config.h
    "$comp" main.cpp -o ./run-P2--4-3-1-2.o $flags

fi

# Compile all executables for P2
if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P2 ..."
    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P3--1-2-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P3--1-3-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P3--2-1-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P3--2-3-1-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P3--3-1-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P3--3-2-1-4.o $flags


    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"12"/" config.h
    "$comp" main.cpp -o ./run-P3--1-3-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"13"/" config.h
    "$comp" main.cpp -o ./run-P3--1-4-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"14"/" config.h
    "$comp" main.cpp -o ./run-P3--3-1-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"15"/" config.h
    "$comp" main.cpp -o ./run-P3--3-4-1-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"16"/" config.h
    "$comp" main.cpp -o ./run-P3--4-1-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"17"/" config.h
    "$comp" main.cpp -o ./run-P3--4-3-1-2.o $flags



    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"18"/" config.h
    "$comp" main.cpp -o ./run-P3--2-3-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"19"/" config.h
    "$comp" main.cpp -o ./run-P3--2-4-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"20"/" config.h
    "$comp" main.cpp -o ./run-P3--3-2-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"21"/" config.h
    "$comp" main.cpp -o ./run-P3--3-4-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"22"/" config.h
    "$comp" main.cpp -o ./run-P3--4-2-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"23"/" config.h
    "$comp" main.cpp -o ./run-P3--4-3-2-1.o $flags


    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"6"/" config.h
    "$comp" main.cpp -o ./run-P3--1-2-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"7"/" config.h
    "$comp" main.cpp -o ./run-P3--1-4-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"8"/" config.h
    "$comp" main.cpp -o ./run-P3--2-1-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"9"/" config.h
    "$comp" main.cpp -o ./run-P3--2-4-1-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"10"/" config.h
    "$comp" main.cpp -o ./run-P3--4-1-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"11"/" config.h
    "$comp" main.cpp -o ./run-P3--4-2-1-3.o $flags

fi

# Compile all executables for P3
if [ "$O_PARTY" = "3" ] || [ "$O_PARTY" = "all" ]
then
    echo "Compiling executables for P3 ..."
    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"6"/" config.h
    "$comp" main.cpp -o ./run-P4--1-2-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"7"/" config.h
    "$comp" main.cpp -o ./run-P4--1-4-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"8"/" config.h
    "$comp" main.cpp -o ./run-P4--2-1-4-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"9"/" config.h
    "$comp" main.cpp -o ./run-P4--2-4-1-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"10"/" config.h
    "$comp" main.cpp -o ./run-P4--4-1-2-3.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"11"/" config.h
    "$comp" main.cpp -o ./run-P4--4-2-1-3.o $flags



    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"12"/" config.h
    "$comp" main.cpp -o ./run-P4--1-3-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"13"/" config.h
    "$comp" main.cpp -o ./run-P4--1-4-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"14"/" config.h
    "$comp" main.cpp -o ./run-P4--3-1-4-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"15"/" config.h
    "$comp" main.cpp -o ./run-P4--3-4-1-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"16"/" config.h
    "$comp" main.cpp -o ./run-P4--4-1-3-2.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"17"/" config.h
    "$comp" main.cpp -o ./run-P4--4-3-1-2.o $flags



    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"18"/" config.h
    "$comp" main.cpp -o ./run-P4--2-3-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"19"/" config.h
    "$comp" main.cpp -o ./run-P4--2-4-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"2"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"20"/" config.h
    "$comp" main.cpp -o ./run-P4--3-2-4-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"1"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"21"/" config.h
    "$comp" main.cpp -o ./run-P4--3-4-2-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"22"/" config.h
    "$comp" main.cpp -o ./run-P4--4-2-3-1.o $flags

    sed -i -e "s/\(PARTY \).*/\1"0"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"23"/" config.h
    "$comp" main.cpp -o ./run-P4--4-3-2-1.o $flags


        sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h
    "$comp" main.cpp -o ./run-P4--1-2-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"1"/" config.h
    "$comp" main.cpp -o ./run-P4--1-3-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"2"/" config.h
    "$comp" main.cpp -o ./run-P4--2-1-3-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"3"/" config.h
    "$comp" main.cpp -o ./run-P4--2-3-1-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"4"/" config.h
    "$comp" main.cpp -o ./run-P4--3-1-2-4.o $flags

    sed -i -e "s/\(PARTY \).*/\1"3"/" config.h
    sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"5"/" config.h
    "$comp" main.cpp -o ./run-P4--3-2-1-4.o $flags

fi

sed -i -e "s/\(SPLIT_ROLES_OFFSET \).*/\1"0"/" config.h

echo "Finished compiling"
