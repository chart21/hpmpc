#!/bin/bash
helpFunction()
{
   echo "Script to compile and run 6 mixed constellations of players in parallel"
   echo -e "\t-p Party number or all for running locally"
   echo -e "\t-a IP address of lower index player "
   echo -e "\t-b IP address of higher index player "
   echo -e "\t-x Compiler (g++/clang++/..)"
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:x:" opt
do
   case "$opt" in
      p ) PARTY="$OPTARG" ;;
      a ) IP1="$OPTARG" ;;
      b ) IP2="$OPTARG" ;;
      x ) COMPILER="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

comp="g++"
if [ ! -z "$COMPILER" ]
then
comp="$COMPILER"
fi

flags="-march=native -Ofast -std=c++2a -pthread -lssl -lcrypto"


for i in {0..2}
    do
    for j in {0..2}
        do
            for z in {0..1}
                do
                    let "s = (1-z)*(j+i)%3+z*(3+2-i-j)%3"
                    # s= $(((1-z)*(j+i)%3+z*(2-i-j)%3))
                    echo $s
            if [ "$i" = "$PARTY" ] || [ "$PARTY" = "all" ];
            then
                sed -i -e "s/\(PARTY \).*/\1"$s"/" config.h
                sed -i -e "s/\(BASE_PORT \).*/\1"$((6000 + (j+z*3) * 1000))"/" config.h
                if [ "$LIVE" = "0" ] && [ "$INIT" = "1" ]; 
                then
                    sed -i -e "s/\(LIVE \).*/\10/" config.h
                    sed -i -e "s/\(INIT \).*/\11/" config.h
                    echo "Compiling INIT executable for P-"$s"-"$i"-"$z" ..."
                    "$comp" tmain.cpp -o ./search-P"$s"-"$i"-"$z"-INIT.o $flags
                    ./search-P"$j"-"$i"-"$z"-INIT.o
                    sed -i -e "s/\(LIVE \).*/\11/" config.h
                    sed -i -e "s/\(INIT \).*/\10/" config.h
                fi
                    echo "Compiling executable for P-"$s"-"$i"-"$z" ..."
                    "$comp" tmain.cpp -o ./search-P"$s"-"$i"-"$z".o $flags
            fi
        done
    done
done
echo "Finished compiling"
