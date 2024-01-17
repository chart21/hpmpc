#!/bin/bash
functions=(70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 100 101 102)
packbool=(0 1)
protocols=(5)
pre=(0 1)
for pro in "${protocols[@]}"
do
    for f in "${functions[@]}"
    do
        for pa in "${packbool[@]}"
        do
            for pr in "${pre[@]}"
            do
                    ./scripts/config.sh -f $f -p all3 -c $pa -e $pre -s $pro
                    echo "Running function $f with protocol, $pro, and packbool, $pa, and pre, $pr"
                    ./scripts/run_locally.sh -n 3
            done
        done
    done
done


