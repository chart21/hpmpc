#!/bin/bash
# shellcheck disable=SC1091,2154

#
# Script is run locally on experiment server.
#

# exit on error
set -e
# log every command
set -x

# load global variables
REPO_DIR=$(pos_get_variable repo_hpmpc_dir --from-global) # hpmpc
REPO2_DIR=$(pos_get_variable repo_dir --from-global) # mpcbench
REPO3_DIR=$(pos_get_variable repo_mpspdz_dir --from-global) # mpspdz
manipulate=$(pos_get_variable manipulate --from-global)
# load loop variables/switches
size=$(pos_get_variable input_size --from-loop)
protocol=$(pos_get_variable protocol --from-loop)
datatype=$(pos_get_variable datatype --from-loop)
preprocess=$(pos_get_variable preprocess --from-loop)
splitroles=$(pos_get_variable splitroles --from-loop)
packbool=$(pos_get_variable packbool --from-loop)
optshare=$(pos_get_variable optshare --from-loop)
ssl=$(pos_get_variable ssl --from-loop)
threads=$(pos_get_variable threads --from-loop)
fun=$(pos_get_variable function --from-loop)
txbuffer=$(pos_get_variable txbuffer --from-loop)
rxbuffer=$(pos_get_variable rxbuffer --from-loop)
verifybuffer=$(pos_get_variable verifybuffer --from-loop)
comp="g++-12"

timerf="%M (Maximum resident set size in kbytes)\n\
%e (Elapsed wall clock time in seconds)\n\
%P (Percent of CPU this job got)"
player=$1
environ=""
# test types to simulate changing environments like cpu frequency or network latency
read -r -a types <<< "$2"
network=10.10."$3"
partysize=$4
# experiment type to allow small differences in experiments
etype=$5
touch testresults

SCHEDULE_PATH="$REPO_DIR/MP-SPDZ/Schedules"
BYTECODE_PATH="$REPO_DIR/MP-SPDZ/Bytecodes"
FUNC_PATH="$REPO_DIR/MP-SPDZ/Functions/bench"

cp "$FUNC_PATH"/* "$REPO3_DIR/Programs/Source"

files=($(find "$FUNC_PATH" -type f -exec basename {} .mpc \;))

mkdir -p "$SCHEDULE_PATH" "$BYTECODE_PATH"

cd "$REPO3_DIR"

domain="-R"
bitlength=32

# map function number to function name
case "$fun" in
    501) func="custom" ;;
    506) func="Int_Multiplication" ;;
    507|508|509) func="Int_Compare" ;;
    510) func="Int_Division" ;;
    511) func="Input" ;;
    512) func="Reveal" ;;
    513|514|515) func="SecureMax" ;;
    516|517|518) func="SecureMin" ;;
    519) func="SecureMean" ;;
    520) func="PrivateSetIntersection" ;;
    521|522|523) func="SecureAuction" ;;
    524)
        domain="-B"
        bitlength=1
        func="BIT_AND" ;;
    525)
        domain="-B"
        bitlength=128
        func="AES" ;;
    526|527|528) func="LogReg" ;;
    529|530|531) func="LeNet" ;;
    532|533|534) func="VGG" ;;
    *)
        func="Int_Multiplication"
        echo WARNING: "$fun": unkown function defaulting to Int_Multiplication
esac

echo "./compile.py --budget 200000 -l $domain $bitlength -K LTZ,EQZ $func $size"
./compile.py --budget 200000 -l "$domain" "$bitlength" -K LTZ,EQZ "$func" "$size"
mv "$REPO3_DIR"/Programs/Schedules/"$func"-"$size".sch "$SCHEDULE_PATH"/"$func".sch

mv ./Programs/Bytecode/* "$BYTECODE_PATH"/

binary_in="$REPO3_DIR"/Player-Data/Input-Binary-P0-0
if [[ -e "$binary_in" ]]; then
    for i in {0..15}; do
        cp "$binary_in" "$REPO_DIR"/MP-SPDZ/Input/Input-Binary-P0-0-"$i"
    done
fi

cd "$REPO_DIR"

# different split role script, different ip definition...
if [ "$splitroles" -lt 1 ]; then
    # define ip addresses of the other party members
    if [ "$protocol" -lt 7 ]; then
        [ "$player" -eq 0 ] && ipA="$network".3 && ipB="$network".4
        [ "$player" -eq 1 ] && ipA="$network".2 && ipB="$network".4
        [ "$player" -eq 2 ] && ipA="$network".2 && ipB="$network".3
    else
        [ "$player" -eq 0 ] && ipA="$network".3 && ipB="$network".4 && ipC="$network".5
        [ "$player" -eq 1 ] && ipA="$network".2 && ipB="$network".4 && ipC="$network".5
        [ "$player" -eq 2 ] && ipA="$network".2 && ipB="$network".3 && ipC="$network".5
        [ "$player" -eq 3 ] && ipA="$network".2 && ipB="$network".3 && ipC="$network".4
    fi
else
    # define all ips
    ipA="$network".2
    ipB="$network".3
    ipC="$network".4
    ipD="$network".5
fi

{
    echo "./scripts/config.sh -p $player -n $size -d $datatype -s $protocol -e $preprocess -h $ssl -x $comp"

    # set config and compile experiment
    if [ "$splitroles" -eq 0 ]; then
        /bin/time -f "$timerf" ./scripts/config.sh -p "$player" -n "$size" -d "$datatype" -x "$comp" \
            -s "$protocol" -e "$preprocess" -c "$packbool" -o "$optshare" -h "$ssl" -b 25000 \
            -j "$threads" -f "$fun" -y "$txbuffer" -z "$rxbuffer" -m "$verifybuffer"
    else
        # with splitroles active, "-p 3" would through error. Omit -p as unneeded
        /bin/time -f "$timerf" ./scripts/config.sh -n "$size" -d "$datatype" -x "$comp" \
            -s "$protocol" -e "$preprocess" -c "$packbool" -o "$optshare" -h "$ssl" -b 25000 \
            -j "$threads" -f "$fun" -y "$txbuffer" -z "$rxbuffer" -m "$verifybuffer"
    fi
    
    [ "$splitroles" -eq 1 ] && ./scripts/split-roles-3-compile.sh -p "$player" -a "$ipA" -b "$ipB" -c "$ipC" -x "$comp"
    [ "$splitroles" -eq 2 ] && ./scripts/split-roles-3to4-compile.sh -p "$player" -a "$ipA" -b "$ipB" -c "$ipC" -d "$ipD" -x "$comp"
    [ "$splitroles" -eq 3 ] && ./scripts/split-roles-4-compile.sh -p "$player" -a "$ipA" -b "$ipB" -c "$ipC" -d "$ipD" -x "$comp"
    
    echo "$(du -BM run-P* | cut -d 'M' -f 1 | head -n 1) (Binary file size in MiB)"

} |& tee testresults

echo -e "\n========\n" >> testresults

####
#  environment manipulation section start
####
# shellcheck source=../host_scripts/manipulate.sh
source "$REPO2_DIR"/host_scripts/hpmpc_mp-spdz/manipulate.sh

if [[ "${types[*]}" == *" LATENCY=0 "* ]]; then
    types=("${types[@]/LATENCY}")
fi

case " ${types[*]} " in
    *" CPUS "*)
        limitCPUs;;&
    *" RAM "*)
        limitRAM;;&
    *" QUOTAS "*)
        setQuota;;&
    *" FREQS "*)
        setFrequency;;&
    *" BANDWIDTHS "*)
        # check whether to manipulate a combination
        case " ${types[*]} " in
            *" LATENCIES "*)
            case " ${types[*]} " in
                *" PACKETDROPS "*)
                    setAllParameters "$partysize";;
                *)
                setLatencyBandwidth;;
            esac;;                 
            *" PACKETDROPS "*) # a.k.a. packet loss
                setBandwidthPacketdrop;;
            *)
                limitBandwidth;;
        esac;;
    *" LATENCIES "*)
        if [[ " ${types[*]} " == *" PACKETDROPS "* ]]; then
            setPacketdropLatency
        else
            setLatency
        fi;;
    *" PACKETDROPS "*)
        setPacketdrop;;
esac
####
#  environment manipulation section stop
####

success=true

pos_sync --timeout 600

# run the SMC protocol
                              # skip 4th node here
if [ "$splitroles" -eq 0 ]; then 
    if [ "$protocol" -lt 7 ]; then
        if [ "$player" -lt 3 ]; then
            /bin/time -f "$timerf" timeout 1000s ./run-P"$player".o "$ipA" "$ipB" &>> testresults || success=false
        fi
    else
        /bin/time -f "$timerf" timeout 1000s ./run-P"$player".o "$ipA" "$ipB" "$ipC" &>> testresults || success=false
    fi
elif [ "$splitroles" -eq 1 ] && [ "$player" -lt 3 ]; then
    /bin/time -f "$timerf" timeout 1000s ./scripts/split-roles-3-execute.sh -p "$player" -a "$ipA" -b "$ipB" -c "$ipC" &>> testresults || success=false
elif [ "$splitroles" -eq 2 ]; then
    /bin/time -f "$timerf" timeout 1000s ./scripts/split-roles-3to4-execute.sh -p "$player" -a "$ipA" -b "$ipB" -c "$ipC" -d "$ipD" &>> testresults || success=false
elif [ "$splitroles" -eq 3 ]; then
    /bin/time -f "$timerf" timeout 1000s ./scripts/split-roles-4-execute.sh -p "$player" -a "$ipA" -b "$ipB" -c "$ipC" -d "$ipD" &>> testresults || success=false
fi

# divide external runtime x*j
# Todo: divide normal binary run by j*j


    # binary:   calculate mean of j results running concurrent ( /j *j )
    # 3nodes:   calculate mean of 6*j results running concurrent ( /6*j *6*j )
    # 3-4nodes: calculate mean of 24*j results running concurrent ( /24*j *24*j )
    # 4nodes:   calculate mean of 24*j results running concurrent ( /24*j *24*j )
#default divisor
divisor=1
divisorExt=1
    
    [ "$splitroles" -eq 0 ] && divisor=$((threads*threads)) && divisorExt=$((threads))
    [ "$splitroles" -eq 1 ] && divisor=$((6*6*threads*threads)) && divisorExt=$((6*threads))
    [ "$splitroles" -eq 2 ] && divisor=$((24*24*threads*threads)) && divisorExt=$((24*threads))
    [ "$splitroles" -eq 3 ] && divisor=$((24*24*threads*threads)) && divisorExt=$((24*threads))

    # sum=$(grep "measured to initialize program" testresults | cut -d 's' -f 2 | awk '{print $5}' | paste -s -d+ | bc)
    # average=$(echo "scale=6;$sum / $divisor" | bc -l)
    # echo "Time measured to initialize program: ${average}s" &>> testresults
       max=$(grep "measured to initialize program" testresults | cut -d 's' -f 2 | awk '{print $5}' | sort -nr | head -1) 
    average=$(echo "scale=6;$max / $divisorExt" | bc -l)
    echo "Time measured to initialize program: ${average}s" &>> testresults

    if [ "$preprocess" -eq 1 ]; then
        # sum=$(grep "preprocessing chrono" testresults | cut -d 's' -f 4 | awk '{print $3}' | paste -s -d+ | bc)
        # average=$(echo "scale=6;$sum / $divisor" | bc -l)
    # echo "Time measured to perform preprocessing chrono: ${average}s" &>> testresults
    max=$(grep "preprocessing chrono" testresults | cut -d 's' -f 4 | awk '{print $3}' | sort -nr | head -1) 
        average=$(echo "scale=6;$max / $divisorExt" | bc -l)
    echo "Time measured to perform preprocessing chrono: ${average}s" &>> testresults
    fi

    sum=$(grep "computation clock" testresults | cut -d 's' -f 2 | awk '{print $6}' | paste -s -d+ | bc)
    average=$(echo "scale=6;$sum / $divisor" | bc -l)
    echo "Time measured to perform computation clock: ${average}s" &>> testresults

    sum=$(grep "computation getTime" testresults | cut -d 's' -f 2 | awk '{print $6}' | paste -s -d+ | bc)
    average=$(echo "scale=6;$sum / $divisor" | bc -l)
    echo "Time measured to perform computation getTime: ${average}s" &>> testresults

max=$(grep "computation chrono" testresults | cut -d 's' -f 2 | awk '{print $6}' | sort -nr | head -1)
    average=$(echo "scale=6;$max / $divisorExt" | bc -l)
    echo "Time measured to perform computation chrono: ${average}s" &>> testresults
# sum=$(grep "computation chrono" testresults | cut -d 's' -f 2 | awk '{print $6}' | paste -s -d+ | bc)
    # average=$(echo "scale=6;$sum / $divisor" | bc -l)
    # echo "Time measured to perform computation chrono: ${average}s" &>> testresults

    runtimeext=$(grep "Elapsed wall clock" testresults | tail -n 1 | cut -d ' ' -f 1)
    average=$(echo "scale=6;$runtimeext / $divisorExt" | bc -l)
    echo "$average (Elapsed wall clock time in seconds)" &>> testresults


pos_sync

####
#  environment manipulation reset section start
####

case " ${types[*]} " in

    *" FREQS "*)
        resetFrequency;;&
    *" RAM "*)
        unlimitRAM;;&
    *" BANDWIDTHS "*|*" LATENCIES "*|*" PACKETDROPS "*)
    	resetTrafficControl "$partysize";;&
    *" CPUS "*)
        unlimitCPUs
esac

####
#  environment manipulation reset section stop
####

echo "experiment finished"  >> testresults
pos_upload --loop testresults
pos_upload --loop terminal_output.txt
# abort if no success
$success
