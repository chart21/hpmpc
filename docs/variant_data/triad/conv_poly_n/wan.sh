#!/bin/bash
# wan.sh REPS OUT LAT BW (coinbase; LAT -1 BW -1: LAN): WAN (20 ms, 200 Mbit/s per direction, shape_network_alt.sh) runs of the list_wan builds with
# the packed convs at N = 4096 / 8192 (CONV_POLY_N); the shaping is reset (-1 / -1) at the end and on any exit
R=$1 OUT=$2 LAT=${3:-20} BW=${4:-200}
SH=/root/hpmpc/measurements/network_shaping/shape_network_alt.sh
reset() { for n in flare polynize; do ssh $n "LATENCY_MS=-1 BANDWIDTH_MBIT=-1 bash $SH" > /dev/null 2>&1; ssh $n "tc qdisc show | grep -c netem" | sed "s/^/reset $n netem qdiscs: /"; done; }
trap reset EXIT
for n in flare polynize; do ssh $n "LATENCY_MS=$LAT BANDWIDTH_MBIT=$BW bash $SH" | tail -1; done
ssh flare "ping -c 3 -q 10.0.4.1 | tail -1"
[ -f $OUT ] || echo "pair,name,tag,wall,pre,online,acc,tot,hash,np,phases,live,mb_pre,mb_live,waited,err,conv_line" > $OUT
for r in $(seq 1 $R); do
  for N in $(awk '{print $1}' /tmp/wan/list_wan.txt); do for P in 4096 8192; do
    ssh flare "pkill -f '^/root/vx/'"; ssh polynize "pkill -f '^/root/vx/'"; sleep 1
    T=${N}_n${P}_l${LAT}_r$r
    row=$(bash /tmp/mx/vrun.sh fp $N $T "CONV_POLY_N=$P")
    cl=$(ssh flare "grep -a 'TRIPLE_STATS (Aggregated)-- CONV' /root/vx/$T.out0 | head -1 | sed -E 's/.*s PRE: ([0-9.]+).*MB SENT PRE: ([0-9.]+).*MB RECEIVED PRE: ([0-9.]+).*/\1;\2;\3/'")
    echo "$row,$cl" >> $OUT; echo "$row,$cl"
  done; done
done
echo WAN_DONE
