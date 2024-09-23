export ITERATIONS=1 # replace with number of repeated executions you want to run
export IP0=10.10.94.2 # replace with your IPs
export IP1=10.10.94.3
export IP2=10.10.94.3
export IP3=10.10.94.3
export PID=0 # replace with node id
export DATTYPE=256 # replace with highest DATTYPE supported by your hardware
export DATTYPES=1,8,16,32,64,128,256 # only include DATTYPES supported by your hardware
export REDUCED="PROCESS_NUM=1 DATTYPE=32"


#4PC

#Figure 1

#Apply Bandwidth
#100,200,500,1000,2000,4000

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=9 DATTYPE=$DATTYPE $REDUCED

#Reset network shaping

#Figure 10

#Apply Latency
#1,3,5,7,9,11ms
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/aes_latency.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPES PROTOCOL=9,10,12 $REDUCED
#Reset network shaping

#Apply Bandwidth
#1000,2000,4000,6000,8000,10000 Mbps
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/mult_throughput.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPE $REDUCED
#Reset network shaping

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/vector_matrix.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPE $REDUCED


#Figure 1
#Apply Bandwidth
#100,200,500,1000,2000,4000

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure11/runtime_baseline -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=9 DATTYPE=$DATTYPE $REDUCED

#Reset network shaping

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure11/runtime_Trio_Quad -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=12 DATTYPE=$DATTYPE $REDUCED

#Figure 9
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure9/bits_per_register.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPES $REDUCED

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure9/num_processes.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3-p $PID --override DATTYPE=$DATTYPE $REDUCED

#Table 10

declare -a t10_settings=("CMAN.json" "WAN1.json" "Mixed.json")
for setting in ${t10_settings[@]}
do
measurements/configs/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override -l 2 -f measurements/configs/network_shaping/$setting


python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID $REDUCED

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID $REDUCED

done
#Reset network shaping

#Table 6
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override PROTOCOL=8,9,10,12 DATTYPE=$DATTYPE $REDUCED

#Table 7
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/throughput_4PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/throughput_4PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Table 8
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table8/throughput_4PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table8/throughput_4PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Table 9

#Apply WAN1 for all if REDUCED is not setto ""
if [ -z "$REDUCED" ]
then
    python3 measurements/configs/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override -l 2 -f measurements/configs/network_shaping/WAN1.json
fi

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_bdw/aes-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_bdw/aes-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_lat/aes1-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_lat/aes1-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/vector_prod20k/dot-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/vector_prod20k/dot-PRE-4PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPE $REDUCED

#Reset network shaping

#3PC

#Figure 1
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure1 -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2 DATTYPE=$DATTYPE $REDUCED

#Figure 10
#Apply Latency
#1,3,5,7,9,11ms

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure10/aes_latency.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override DATTYPE=$DATTYPES PROTOCOL=2,5 $REDUCED
#Reset network shaping

#Figure 11
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure11/runtime_baseline -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2 DATTYPE=$DATTYPE $REDUCED
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/figure11/runtime_Trio_Quad -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=5 DATTYPE=$DATTYPE $REDUCED

#Figure 9
#-

#Table 10

for setting in ${t10_settings[@]}
do
measurements/configs/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override -l 2 -f measurements/configs/network_shaping/$setting


python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID $REDUCED

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID $REDUCED

done
#Reset network shaping

#Table 6
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table6/ -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override PROTOCOL=2,3,5 DATTYPE=$DATTYPE $REDUCED

#Table 7
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/throughput_3PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table7/throughput_3PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE

#Table 8
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table8/throughput_3PC_PRE0.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table8/throughput_3PC_PRE1.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2 -p $PID --override DATTYPE=$DATTYPE

#Table 9

#Apply WAN1 for all if REDUCED is not setto ""

if [ -z "$REDUCED" ]
then
    python3 measurements/configs/network_shaping/shape_network.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override -l 2 -f measurements/configs/network_shaping/WAN1.json
fi

measurements/configs/network_shaping/shape_network.sh - a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID --override -l 2 -f measurements/configs/network_shaping/WAN1.json

python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_bdw/aes-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_bdw/aes-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_lat/aes1-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/aes_lat/aes1-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/vector_prod20k/dot-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE
python3 measurements/run_config.py measurements/configs/artifacts/hpmpc/table10/vector_prod20k/dot-PRE-3PC.conf -i $ITERATIONS -a $IP0 -b $IP1 -c $IP2  -p $PID --override DATTYPE=$DATTYPE

#Reset network shaping
