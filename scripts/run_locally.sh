#!/bin/bash
helpFunction()
{
   echo "Script to run n players locally after compiling."
   echo -e "\t-f Executable to run"
   echo -e "\t-n num_players"
   exit 1 # Exit script after printing help
}

while getopts "f:n:" opt
do
   case "$opt" in
      f ) FUNCTION="$OPTARG" ;;
      n ) NUM_PLAYERS="$OPTARG" ;;
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

for (( i=0; i<$NUM_PLAYERS; i++ ))
do
    ./executables/"$FUNCTION"-P"$i".o &
done

wait
