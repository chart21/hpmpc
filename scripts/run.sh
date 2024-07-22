#!/bin/bash
helpFunction()
{
   echo "Script to run a n-PC protocol in a distributed setting"
   echo -e "\t-p Party number"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-d IP address of player 3 (if ip matches player_id can be empty)"
   echo -e "\t-n Number of players"
   echo -e "\t-g Number of GPUs to use"
   echo -e "\t-s SplitRoles Identifier"

   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:n:g:s:" opt
do
   case "$opt" in
      p ) O_PARTY="$OPTARG" ;;
      a ) IP1="$OPTARG" ;;
      b ) IP2="$OPTARG" ;;
      c ) IP3="$OPTARG" ;;
      d ) IP4="$OPTARG" ;;
      n ) NUM_PLAYERS="$OPTARG" ;;
      g ) NUM_GPUS="$OPTARG" ;;
      s ) SPLIT_ROLES="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


O_IP1="127.0.0.1"
O_IP2="127.0.0.1"
O_IP3="127.0.0.1"
O_IP4="127.0.0.1"
O_NUM_PLAYERS=3
O_NUM_GPUS=0
O_SPLIT_ROLES=0

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

if [ ! -z "$NUM_PLAYERS" ];
then
O_NUM_PLAYERS="$NUM_PLAYERS"
fi

if [ ! -z "$NUM_GPUS" ];
then
O_NUM_GPUS="$NUM_GPUS"
fi

if [ ! -z "$SPLIT_ROLES" ];
then
O_SPLIT_ROLES="$SPLIT_ROLES"
fi

#if O_SPLIT_ROLES is 0, then run the protocol with 3 players
if [ "$O_SPLIT_ROLES" = "0" ]; then
    if [ "$O_PARTY" = "all" ]
    then
        scripts/run_locally.sh -n $O_NUM_PLAYERS
    else
        scripts/run_distributed.sh -p $O_PARTY -a $O_IP1 -b $O_IP2 -c $O_IP3 -d $O_IP4
    fi
elif [ "$O_SPLIT_ROLES" = "1" ]; then
    scripts/split-roles-3-execute.sh -p $O_PARTY -a $O_IP1 -b $O_IP2 -c $O_IP3 -g $O_NUM_GPUS
elif [ "$O_SPLIT_ROLES" = "2" ]; then
    scripts/split-roles-3to4-execute.sh -p $O_PARTY -a $O_IP1 -b $O_IP2 -c $O_IP3 -d $O_IP4
elif [ "$O_SPLIT_ROLES" = "3" ]; then
    scripts/split-roles-4-execute.sh -p $O_PARTY -a $O_IP1 -b $O_IP2 -c $O_IP3 -d $O_IP4
fi

