#!/bin/bash

helpFunction()
{
   echo "Script to run a n-PC protocol in a distributed or local setting."
   echo ""
   echo "Usage: run.sh -p <party> [options]"
   echo ""
   echo "Options:"
   echo -e "\t-p <party>       Party number, or 'all' to run all parties locally"
   echo -e "\t-a <ip>          IP address of player 0"
   echo -e "\t-b <ip>          IP address of player 1"
   echo -e "\t-c <ip>          IP address of player 2"
   echo -e "\t-d <ip>          IP address of player 3"
   echo -e "\t-n <num>         Number of players"
   echo -e "\t-g <num>         Number of GPUs to use (SplitRoles)"
   echo -e "\t-s <id>          SplitRoles identifier"
   echo -e "\t-G <player:dev>  Assign a specific CUDA device to a player (local runs only)."
   echo -e "\t                 Can be repeated for multiple players."
   echo -e "\t                 Players without -G get all GPUs (no restriction)."
   echo -e "\t                 Examples:"
   echo -e "\t                   -G 0:0 -G 1:1   → P0 uses GPU 0, P1 uses GPU 1"
   echo -e "\t                   -G 1:2           → only P1 restricted to GPU 2"
   echo -e "\t                   (no -G)          → all players see all GPUs"
   exit 1
}

GPU_ARGS=()

while getopts "p:a:b:c:d:n:g:s:G:h" opt
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
      G ) GPU_ARGS+=("-G" "$OPTARG") ;;
      h ) helpFunction ;;
      ? ) helpFunction ;;
   esac
done

O_IP1="127.0.0.1"
O_IP2="127.0.0.1"
O_IP3="127.0.0.1"
O_IP4="127.0.0.1"
O_NUM_PLAYERS=3
O_NUM_GPUS=0
O_SPLIT_ROLES=0

[ -n "$IP1" ]        && O_IP1="$IP1"
[ -n "$IP2" ]        && O_IP2="$IP2"
[ -n "$IP3" ]        && O_IP3="$IP3"
[ -n "$IP4" ]        && O_IP4="$IP4"
[ -n "$NUM_PLAYERS" ] && O_NUM_PLAYERS="$NUM_PLAYERS"
[ -n "$NUM_GPUS" ]   && O_NUM_GPUS="$NUM_GPUS"
[ -n "$SPLIT_ROLES" ] && O_SPLIT_ROLES="$SPLIT_ROLES"

if [ "$O_SPLIT_ROLES" = "0" ]; then
    if [ "$O_PARTY" = "all" ]
    then
        scripts/run_locally.sh -n $O_NUM_PLAYERS "${GPU_ARGS[@]}"
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
