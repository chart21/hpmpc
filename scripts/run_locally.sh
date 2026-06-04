#!/bin/bash
helpFunction()
{
   echo "Script to run n players locally after compiling."
   echo ""
   echo "Usage: run_locally.sh [-f executable] -n num_players [-G player:device] ..."
   echo ""
   echo "Options:"
   echo -e "\t-f <name>        Executable prefix to run (default: run)"
   echo -e "\t-n <num>         Number of players"
   echo -e "\t-G <player:dev>  Assign a specific CUDA device to a player."
   echo -e "\t                 Can be repeated for multiple players."
   echo -e "\t                 Players without -G get all GPUs (no restriction)."
   echo -e "\t                 Examples:"
   echo -e "\t                   -G 0:0 -G 1:1   → P0 uses GPU 0, P1 uses GPU 1"
   echo -e "\t                   -G 1:2           → only P1 restricted to GPU 2"
   echo -e "\t                   (no -G)          → all players see all GPUs"
   exit 1
}

FUNCTION="run"
NUM_PLAYERS=""
declare -A GPU_MAP   # GPU_MAP[player]=device

while getopts "f:n:G:h" opt
do
   case "$opt" in
      f ) FUNCTION="$OPTARG" ;;
      n ) NUM_PLAYERS="$OPTARG" ;;
      G )
         PLAYER="${OPTARG%%:*}"
         DEVICE="${OPTARG##*:}"
         if [ "$PLAYER" = "$OPTARG" ] || [ -z "$PLAYER" ] || [ -z "$DEVICE" ]; then
             echo "❌ -G requires format player:device (e.g. -G 0:0)"
             exit 1
         fi
         GPU_MAP[$PLAYER]="$DEVICE"
         ;;
      h ) helpFunction ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$NUM_PLAYERS" ]
then
    echo "Please specify number of players with -n"
    exit 1
fi

for (( i=0; i<$NUM_PLAYERS; i++ ))
do
    if [ -n "${GPU_MAP[$i]+set}" ]; then
        echo "🎮 P$i → CUDA_VISIBLE_DEVICES=${GPU_MAP[$i]}"
        CUDA_VISIBLE_DEVICES="${GPU_MAP[$i]}" ./executables/"$FUNCTION"-P"$i".o &
    else
        ./executables/"$FUNCTION"-P"$i".o &
    fi
done

wait
