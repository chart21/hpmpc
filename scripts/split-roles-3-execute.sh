#!/bin/bash
helpFunction()
{
   echo "Script to run 6 mixed constellations of a 3-PC protocol in parallel"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"

   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP0="$OPTARG" ;;
      b ) IP1="$OPTARG" ;;
      c ) IP2="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


O_IP0="127.0.0.1"
O_IP1="127.0.0.1"
O_IP2="127.0.0.1"

if [ ! -z "$IP1" ];
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




# Run all executables for P1
if [ "$O_PARTY" = "0" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P0--0-1-2.o $O_IP1 $O_IP2 &
    ./run-P0--0-2-1.o $O_IP2 $O_IP1 &
    ./run-P0--1-0-2.o $O_IP1 $O_IP2 &
    ./run-P0--1-2-0.o $O_IP1 $O_IP2 &
    ./run-P0--2-0-1.o $O_IP2 $O_IP1 &
    ./run-P0--2-1-0.o $O_IP2 $O_IP1 &
fi
# Run all executables for P2
if [ "$O_PARTY" = "1" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P1--0-1-2.o $O_IP0 $O_IP2 &
    ./run-P1--0-2-1.o $O_IP0 $O_IP2 &
    ./run-P1--1-0-2.o $O_IP0 $O_IP2 &
    ./run-P1--1-2-0.o $O_IP2 $O_IP0 &
    ./run-P1--2-0-1.o $O_IP2 $O_IP0 &
    ./run-P1--2-1-0.o $O_IP2 $O_IP0 &
fi
# Run all executables for P3
if [ "$O_PARTY" = "2" ] || [ "$O_PARTY" = "all" ];
then
    ./run-P2--0-1-2.o $O_IP0 $O_IP1 &
    ./run-P2--0-2-1.o $O_IP0 $O_IP1 &
    ./run-P2--1-0-2.o $O_IP1 $O_IP0 &
    ./run-P2--1-2-0.o $O_IP1 $O_IP0 &
    ./run-P2--2-0-1.o $O_IP0 $O_IP1 &
    ./run-P2--2-1-0.o $O_IP1 $O_IP0 &
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

