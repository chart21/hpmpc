#!/bin/bash
ITERATIONS=1 # replace with number of repeated executions you want to run
IP0=127.0.0.1
IP1=127.0.0.1
IP2=127.0.0.1
IP3=127.0.0.1
PID=all # replace with node id
DATTYPE=256 # replace with highest DATTYPE supported by your hardware
DATTYPES=1,8,16,32,64,256 # only include DATTYPES supported by your hardware
REDUCED="PROCESS_NUM=1 DATTYPE=32 NUM_INPUTS=1"
REDUCED_PROCESS_NUM="1,2,4,8"

helpFunction()
{
   echo "Script to run all tests"
   echo -e "\t-p Party number"
   echo -e "\t-a IP address of player 0 (if ip matches player_id can be empty)"
   echo -e "\t-b IP address of player 1 (if ip matches player_id can be empty)"
   echo -e "\t-c IP address of player 2 (if ip matches player_id can be empty)"
   echo -e "\t-d IP address of player 3 (if ip matches player_id can be empty)"
   echo -e "\t-D Highest DATTYPE supported by hardware"
   echo -e "\t-L List of DATTYPES supported by hardware"
   echo -e "\t-R Reduced settings for faster execution"
   exit 1 # Exit script after printing help
}

while getopts "p:a:b:c:d:D:L:R:i:" opt
do
   case "$opt" in
      p ) PID="$OPTARG" ;;
      a ) IP0="$OPTARG" ;;
      b ) IP1="$OPTARG" ;;
      c ) IP2="$OPTARG" ;;
      d ) IP3="$OPTARG" ;;
      D ) DATTYPE="$OPTARG" ;;
      L ) DATTYPES="$OPTARG" ;;
      R ) REDUCED="$OPTARG" ;;
      i ) ITERATIONS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

echo "=====Starting all measurements with PID $PID, IP0 $IP0, IP1 $IP1, IP2 $IP2, IP3 $IP3, DATTYPE $DATTYPE, DATTYPES $DATTYPES, REDUCED $REDUCED, ITERATIONS $ITERATIONS====="

#4PC

#Figure 1

#Apply Bandwidth
#100,200,500,1000,2000,4000
bandwidths=(100 200 500 1000 2000 4000)
for bandwidth in ${bandwidths[@]}
do
echo "=====Measuring Figure 1 with bandwidth $bandwidth Mbps====="
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -B $bandwidth -L -1

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure1 -f ${bandwidth}Mbps -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=9 DATTYPE=$DATTYPE $REDUCED #rename all .conv files in the directory to remove the bandwidth
done
#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1

#Figure 9
echo "=====Measuring Figure 9====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure9/figure9_bits_per_register.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override $REDUCED DATTYPE=$DATTYPES

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure9/figure9_num_processes.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override $REDUCED PROCESS_NUM=$REDUCED_PROCESS_NUM 

#Figure 10

#Apply Latency
#1,3,5,7,9,11ms
latencies=(1 3 5 7 9 11)
for latency in ${latencies[@]}
do
echo "=====Measuring Figure 10 (a) with latency $latency ms====="


measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -L $latency -B -1

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10a_aes_latency.conf -f ${latency}ms -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE PROTOCOL=9,10,12 $REDUCED

done
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1
#Reset network shaping

#Apply Bandwidth
#1000,2000,4000,6000,8000,10000 Mbps
bandwidths=(1000 2000 4000 6000 8000 10000)
for bandwidth in ${bandwidths[@]}
do
echo "=====Measuring Figure 10 (b) with bandwidth $bandwidth Mbps====="
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -B $bandwidth -L -1

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10b_mult_throughput.conf -f ${bandwidth}Mbps  -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED

done
#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1 

echo "=====Measuring Figure 10 (c)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10c_vector_matrix.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED


#Figure 29
#Apply latencies
latencies=(10 20 30 40 50 60)
for latency in ${latencies[@]}
do
echo "=====Measuring Figure 29 with latency $latency ms====="
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -L $latency -B -1
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure29/figure29_runtime_baseline.conf -f ${latency}ms -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=9 DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure29/figure29_runtime_Trio_Quad.conf -f ${latency}ms  -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 DATTYPE=$DATTYPE $REDUCED
done
#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1



#Table 10

declare -a t10_settings=("CMAN.json" "WAN1.json" "Mixed.json")
for setting in ${t10_settings[@]}
do
echo "=====Measuring Table 10 with network setting $setting====="
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -f measurements/network_shaping/$setting 



python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/table10_aes-4PC.conf -f ${setting} -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override $REDUCED 

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/table10_aes-PRE-4PC.conf -f ${setting} -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override $REDUCED


done
#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1

#Table 6
echo "=====Measuring Table 6====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=8,9,10,12 DATTYPE=$DATTYPE $REDUCED

#Table 7
echo "=====Measuring Table 7====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_4PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_4PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED


#Table 9

#Apply WAN1 for all if REDUCED is not setto ""
if [ -z "$REDUCED" ]
then
    python3 measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -f measurements/network_shaping/WAN1.json
fi

echo "=====Measuring Table 9====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1


echo "=====Finished all 4PC measurements====="
if [ $PID == 3 ]
then
    exit 0
fi

#3PC

#Figure 1
echo "=====Measuring Figure 1 (3PC)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2 DATTYPE=$DATTYPE $REDUCED

#Figure 10
#Apply Latency
#1,3,5,7,9,11ms
latencies=(1 3 5 7 9 11)
for latency in ${latencies[@]}
do
    echo "=====Measuring Figure 10 (a) (3PC) with latency $latency ms====="
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -L $latency -B -1
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/figure10a_aes_latency.conf -f ${latency}ms -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE PROTOCOL=2,5 $REDUCED
done
#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -p $PID -L -1 -B -1


#Figure 11
echo "=====Measuring Figure 11 (3PC)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure29/figure29_runtime_baseline.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2 DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure29/figure29_runtime_Trio_Quad.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 DATTYPE=$DATTYPE $REDUCED

#Figure 9
#-

#Table 10

for setting in ${t10_settings[@]}
do
echo "=====Measuring Table 10 (3PC) with network setting $setting====="
if [ -z "$REDUCED" ]
then
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -f measurements/network_shaping/$setting -f $setting
fi

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/table10_aes-3PC.conf -f $setting -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override $REDUCED

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/table10_aes-PRE-3PC.conf -f $setting -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override $REDUCED

done
#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -L -1 -B -1

#Table 6
echo "=====Measuring Table 6 (3PC)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2,3,5 DATTYPE=$DATTYPE $REDUCED

#Table 7
echo "=====Measuring Table 7 (3PC)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_3PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/table7_throughput_3PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Table 8
echo "=====Measuring Table 8 (3PC)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table8/table8_throughput_3PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table8/table8_throughput_3PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Table 9

#Apply WAN1 for all if REDUCED is not set to ""

if [ -z "$REDUCED" ]
then
    python3 measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -l 2 -f measurements/network_shaping/WAN1.json
fi

echo "=====Measuring Table 9 (3PC)====="
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_bdw/table9_aes-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/aes_lat/table9_aes1-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table9/vector_prod20k/table9_dot-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Reset network shaping
measurements/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override -L -1 -B -1

echo "=====Finished all 3PC measurements====="
