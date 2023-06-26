#!/bin/bash
helpFunction()
{
   echo "Script to compile and run 6 mixed constellations of players in parallel"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of lower index player "
   echo -e "\t-b IP address of higher index player "
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:" opt
do
   case "$opt" in
      p ) PARTY="$OPTARG" ;;
      a ) IP1="$OPTARG" ;;
      b ) IP2="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


if [ "$PARTY" == "all" ]; then
    ./search-P0-0-0.o &
    ./search-P0-1-1.o &
    ./search-P0-2-0.o &
    ./search-P0-0-1.o &
    ./search-P0-1-0.o &
    ./search-P0-2-1.o &
    ./search-P1-0-0.o &
    ./search-P1-1-1.o &
    ./search-P1-2-0.o &
    ./search-P1-0-1.o &
    ./search-P1-1-0.o &
    ./search-P1-2-1.o &
    ./search-P2-0-0.o &
    ./search-P2-1-1.o &
    ./search-P2-2-0.o &
    ./search-P2-0-1.o &
    ./search-P2-1-0.o &
    ./search-P2-2-1.o &
elif [ "$PARTY" == "0" ] || [ "$PARTY" == "2" ]; then
    ./search-P0-"$PARTY"-0.o "$IP1" "$IP2" &
    ./search-P0-"$PARTY"-1.o "$IP2" "$IP1" &
    ./search-P1-"$PARTY"-0.o "$IP2" "$IP1" &
    ./search-P1-"$PARTY"-1.o "$IP1" "$IP2" &
    ./search-P2-"$PARTY"-0.o "$IP1" "$IP2" &
    ./search-P2-"$PARTY"-1.o "$IP2" "$IP1" &
elif [ "$PARTY" == "1" ]; then
    ./search-P0-"$PARTY"-0.o "$IP2" "$IP1" &
    ./search-P0-"$PARTY"-1.o "$IP1" "$IP2" &
    ./search-P1-"$PARTY"-0.o "$IP1" "$IP2" &
    ./search-P1-"$PARTY"-1.o "$IP2" "$IP1" &
    ./search-P2-"$PARTY"-0.o "$IP2" "$IP1" &
    ./search-P2-"$PARTY"-1.o "$IP1" "$IP2" &
fi

FAIL=0
for job in $(jobs -p); do
# echo $job
    wait "$job" || ((++FAIL))
done

# echo $FAIL

if [ "$FAIL" -eq 0 ]; then
    echo "No errors in Split roles ececution"
else
    echo "$FAIL Erros detected in Split roles ececution"
fi

