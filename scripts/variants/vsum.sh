#!/bin/bash
# vsum.sh LOG: pre_max,online_max,acc_correct,acc_total,hash,procs,pre phases (CONV;BN;BOOL;COT;MUX;FC;OTHER s avg),live ms (ACT;CONV;BN),MB pre total,MB live total,rounds,waited
f=$1
pre=$(grep -ao 'preprocessing getTime: [0-9.]*' $f | awk '{if($3>m)m=$3} END{print m+0}')
onl=$(grep -ao 'computation getTime: [0-9.]*' $f | awk '{if($3>m)m=$3} END{print m+0}')
acc=$(grep -ao 'accuracy([0-9]* images): [0-9.]*' $f | awk -F'[(): %]+' '{n+=$2; c+=$2*$4/100} END{printf "%d,%d", c+0.5, n}')
np=$(grep -ac 'computation getTime' $f)
hash=$(for p in $(seq 0 31); do grep -a "PID$p: Output hash" $f | tail -1 | awk '{print $NF}'; done | md5sum | cut -c1-10)
ph=$(grep -a 'TRIPLE_STATS (Aggregated)' $f | sed -E 's/.*Aggregated\)-- ([A-Z]*) .*s PRE: ([0-9.]*).*/\1 \2/' | awk '$1~/^[A-Z]+$/{t[$1]+=$2;n[$1]++} END{printf "%.2f;%.2f;%.2f;%.2f;%.2f;%.2f", t["CONV"]/(n["CONV"]+(n["CONV"]==0)), t["BN"]/(n["BN"]+(n["BN"]==0)), t["BOOL"]/(n["BOOL"]+(n["BOOL"]==0)), t["COT"]/(n["COT"]+(n["COT"]==0)), t["MULTIPLEX"]/(n["MULTIPLEX"]+(n["MULTIPLEX"]==0)), t["FC"]/(n["FC"]+(n["FC"]==0))}')
lv=$(grep -a 'NN_STATS (Aggregated)' $f | sed -E 's/.*Aggregated\)-- ([A-Z0-9]*).*ms LIVE: ([0-9.]*).*/\1 \2/' | awk '{l[$1]+=$2;n[$1]++} END{printf "%.0f;%.0f;%.0f", l["ACTIVATION"]/(n["ACTIVATION"]+(n["ACTIVATION"]==0)), l["CONV2D"]/(n["CONV2D"]+(n["CONV2D"]==0)), l["BATCHNORM2D"]/(n["BATCHNORM2D"]+(n["BATCHNORM2D"]==0))}')
mbp=$(grep -a 'TRIPLE_STATS (Total)' $f | awk '{for(i=1;i<=NF;i++) if($i=="SENT" && $(i+1)=="PRE:") s+=$(i+2)} END{printf "%.0f", s}')
mbl=$(grep -a 'NN_STATS (Aggregated)' $f | sed -E 's/.*MB SENT:([0-9.]*).*/\1/' | awk '{s+=$1} END{printf "%.0f", s}')
wt=$(grep -ao 'waited [0-9.]* s for data in [0-9]* rounds' $f | head -1 | awk '{print $2";"$7}')
err=$(grep -aci 'abort\|misuse\|Segmentation\|terminate called' $f)
echo "$pre,$onl,$acc,$hash,$np,$ph,$lv,$mbp,$wt,$err"
