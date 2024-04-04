#!/bin/bash
#functions=(70 170 270 71 171 271 72 172 272 73 173 273 74 174 274 75 175 275 76 176 276 77 177 277 78 178 278 79 179 279 80 180 280 81 181 281 82 182 282 83 183 283)
functions=(82)
#packbool=(0 1)
#protocols=(12)
#pre=(1)
# loopvals=(4 5)
bitlengths=(32 64 16)
trunc_apporach=(0 1)
trunc_delayed=(0 1)
trunc_then_mult=(0 1)

for f in "${functions[@]}"
    do
    sed -i -e "s/\(define FUNCTION \).*/\1$f/" config.h
    for bitlength in "${bitlengths[@]}"
        do
        sed -i -e "s/\(define BITLENGTH \).*/\1$bitlength/" config.h
        sed -i -e "s/\(define DATTYPE \).*/\1$bitlength/" config.h
        sed -i -e "s/\(define REDUCED_BITLENGTH_k \).*/\1$bitlength/" config.h
        #loop over 1:bitlength-1
        for b in $(seq 1 $((bitlength / 2)))

        # for b in "${loopvals[@]}"
            do
            sed -i -e "s/\(define FRACTIONAL \).*/\1$b/" config.h
            for ta in "${trunc_apporach[@]}"
                do
                sed -i -e "s/\(define TRUNC_APPROACH \).*/\1$ta/" config.h
                for td in "${trunc_delayed[@]}"
                    do
                    sed -i -e "s/\(define TRUNC_DELAYED \).*/\1$td/" config.h
                    for mt in "${trunc_then_mult[@]}"
                        do
                        sed -i -e "s/\(define TRUNC_THEN_MULT \).*/\1$mt/" config.h
                        ./scripts/config.sh -f $f -p all3 
                        echo "Running function $f with bitlength $bitlength, fractional $b, trunc_approach $ta, trunc_delayed $td, trunc_then_mult $mt"
                        ./scripts/run_locally.sh -n 3
                    done
                done
            done
        done
    done
done
    

