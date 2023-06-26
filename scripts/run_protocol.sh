#!/bin/bash
helpFunction()
{
   echo "Script to run 3 players locally after compiling."
   echo -e "\t-f Function to run"
   exit 1 # Exit script after printing help
}

while getopts "f:" opt
do
   case "$opt" in
      f ) FUNCTION="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done
./"$FUNCTION"-P0.o & ./"$FUNCTION"-P1.o &./"$FUNCTION"-P2.o 
