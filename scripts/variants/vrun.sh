#!/bin/bash
# vrun.sh PAIR NAME TAG [ENV]: run variant NAME once on a host pair (fp = flare/polynize, ag = algofi/goracle); prints a CSV row
PAIR=$1 N=$2 TAG=$3 E="$4"
if [ $PAIR = fp ]; then A=flare B=polynize IA=10.0.4.2 IB=10.0.4.1; else A=algofi B=goracle IA=10.0.1.2 IB=10.0.1.1; fi
ssh $A "pkill -x $N.p0" 2>/dev/null; ssh $B "pkill -x $N.p1" 2>/dev/null
T0=$(date +%s.%N)
ssh $A "cd /root/hpmpc && env $E timeout 900 /root/vx/$N.p0 $IA > /root/vx/$TAG.out0 2>&1" &
ssh $B "cd /root/hpmpc && env $E timeout 900 /root/vx/$N.p1 $IB > /root/vx/$TAG.out1 2>&1"
wait
W=$(echo "$(date +%s.%N) - $T0" | bc)
ssh $A "bash /root/vsum.sh /root/vx/$TAG.out0" | sed "s/^/$PAIR,$N,$TAG,$W,/"
