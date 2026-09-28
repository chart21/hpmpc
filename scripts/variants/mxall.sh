#!/bin/bash
# mxall.sh PAIR LIST REPS [FILTER]: run the variants of LIST (built in /root/vx on both nodes) REPS times each,
# appending CSV rows to /tmp/mx/res_<PAIR>.csv (see vsum.sh for the columns)
PAIR=$1 L=$2 R=${3:-1} F=${4:-.}
E="MODEL_DIR=nn/Pygeon/models/pretrained/Cifar_adam_001 MODEL_FILE=ResNet50_avg_CIFAR-10_standard_best.bin DATA_DIR=nn/Pygeon/data/datasets SAMPLES_FILE=CIFAR-10_standard_test_images.bin LABELS_FILE=CIFAR-10_standard_test_labels.bin"
if [ $PAIR = fp ]; then A=flare B=polynize; else A=algofi B=goracle; fi
OUT=/tmp/mx/res_$PAIR.csv
[ -f $OUT ] || echo "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors" > $OUT
for N in $(grep -v '^#' $L | awk '{print $1}' | grep -E "$F"); do
  if ! ssh $A "test -x /root/vx/$N.p0" || ! ssh $B "test -x /root/vx/$N.p1"; then echo "$PAIR,$N,missing" >> $OUT; continue; fi
  for r in $(seq 1 $R); do
    ssh $A "pkill -f '^/root/vx/'" ; ssh $B "pkill -f '^/root/vx/'"; sleep 1
    bash /tmp/mx/vrun.sh $PAIR $N ${N}_r$r "$E" | tee -a $OUT
  done
done
