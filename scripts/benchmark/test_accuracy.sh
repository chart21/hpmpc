#!/bin/bash
cp scripts/benchmark/base_config_accuracy.h config.h 
#Scripts to locally test the accuracy of different networks, truncations approaches, fractional bits, and bitlengths

# functions=(70 170 270 71 171 271 72 172 272 73 173 273 74 174 274 75 175 275 76 176 276 77 177 277 78 178 278 79 179 279 80 180 280 81 181 281 82 182 282 83 183 283)
funcion=(74) # You need a pretrained model and dataset in the respecive folders and specify it in config.txt
protocol=5
num_inputs=100 # How many test images?
bitlengths=(32 64 16)
trunc_apporach=(0 1 2) # 0: Probabilistic, 1: Probabilistic with reduced slack, 2: Deterministic
trunc_delayed=(0 1) # delay truncation until RELU
trunc_then_mult=(0 1) # truncate before or after multiplication
num_repititions=10


sed -i -e "s/\(define PROTOCOL \).*/\1$protocol/" config.h
sed -i -e "s/\(define NUM_INPUTS \).*/\1$num_inputs/" config.h

for f in "${functions[@]}"
    do
    sed -i -e "s/\(define FUNCTION_IDENTIFIER \).*/\1$f/" config.h
    for bitlength in "${bitlengths[@]}"
        do
        sed -i -e "s/\(define BITLENGTH \).*/\1$bitlength/" config.h
        sed -i -e "s/\(define DATTYPE \).*/\1$bitlength/" config.h
        sed -i -e "s/\(define REDUCED_BITLENGTH_k \).*/\1$bitlength/" config.h
        for b in $(seq 1 $((bitlength / 2)))
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
                        for((i=0;i<num_repititions;i++)); do
                                sed -i -e "s/\(define SRNG_SEED \).*/\1$i/" config.h
                        echo "Running function $f with bitlength $bitlength, fractional $b, trunc_approach $ta, trunc_delayed $td, trunc_then_mult $mt"
                        ./scripts/run_locally.sh -n 3
                    done
                done
            done
        done
    done
done
cp scripts/benchmark/base_config.h config.h 
