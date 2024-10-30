#!/bin/bash
helpFunction()
{
   echo "Script to run n players locally after compiling."
   echo -e "\t-f Executable to run"
   echo -e "\t-n num_players"
   echo -e "\t-g Number of GPUs to use"
   exit 1 # Exit script after printing help
}

while getopts "f:n:g:" opt
do
   case "$opt" in
      f ) FUNCTION="$OPTARG" ;;
      n ) NUM_PLAYERS="$OPTARG" ;;
      g ) NUM_GPUS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$FUNCTION" ]
then
    FUNCTION="run"
fi

if [ -z "$NUM_PLAYERS" ]
then
    echo "Please specify number of players with -n"
    exit 1
fi

O_NUM_GPUS=1
if [ ! -z "$NUM_GPUS" ]; then O_NUM_GPUS="$NUM_GPUS"; fi
    if [ $O_NUM_GPUS -gt 1 ]; then
    for (( i=0; i<$NUM_PLAYERS; i++ ))
    do
        CUDA_VISIBLE_DEVICES=$((i % O_NUM_GPUS)) ./executables/"$FUNCTION"-P"$i".o &
    done
else
    for (( i=0; i<$NUM_PLAYERS; i++ ))
    do
        ./executables/"$FUNCTION"-P"$i".o &
    done
fi

wait
