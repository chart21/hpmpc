#!/bin/bash
helpFunction()
{
   echo "Script to run a n-PC protocol in a distributed setting"
   echo -e "\t-p Party number"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-d IP address of player 3 (if ip matches player_id can be empty)"

   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP1="$OPTARG" ;;
      b ) IP2="$OPTARG" ;;
      c ) IP3="$OPTARG" ;;
      d ) IP4="$OPTARG" ;;
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



if [ "$O_PARTY" = "0" ]
then
    ./executables/run-P0.o $O_IP2 $O_IP3 $O_IP4
elif [ "$O_PARTY" = "1" ]
then
    ./executables/run-P1.o $O_IP1 $O_IP3 $O_IP4
elif [ "$O_PARTY" = "2" ]
then
    ./executables/run-P2.o $O_IP1 $O_IP2 $O_IP4
elif [ "$O_PARTY" = "3" ]
then
    ./executables/run-P3.o $O_IP1 $O_IP2 $O_IP3
fi
