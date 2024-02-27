#!/bin/bash
functions=(70 170 270 71 171 271 72 172 272 73 173 273 74 174 274 75 175 275 76 176 276 77 177 277 78 178 278 79 179 279 80 180 280 81 181 281 82 182 282 83 183 283)
packbool=(0 1)
protocols=(12)
pre=(1)
for pro in "${protocols[@]}"
do
    for f in "${functions[@]}"
    do
        for pa in "${packbool[@]}"
        do
            for pr in "${pre[@]}"
            do
                    ./scripts/config.sh -f $f -p all4 -c $pa -e $pr -s $pro
                    echo "Running function $f with protocol, $pro, and packbool, $pa, and pre, $pr"
                    ./scripts/run_locally.sh -n 4
            done
        done
    done
done


