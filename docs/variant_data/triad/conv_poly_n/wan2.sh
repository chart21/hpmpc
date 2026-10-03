#!/bin/bash
# wan2.sh REPS OUT LAT BW (coinbase): the list_wan builds with N = 4096 / 8192 (plain packing, CONV_POLY_N) and with output
# repacking (_rp: N = 8192, special prime, CHEETAH_CONV_REPACK=1), interleaved; shaping reset (-1 / -1) at the end and on exit
R=$1 OUT=$2 LAT=${3:-20} BW=${4:-200}
SH=/root/hpmpc/measurements/network_shaping/shape_network_alt.sh
reset() { for n in flare polynize; do ssh $n "LATENCY_MS=-1 BANDWIDTH_MBIT=-1 bash $SH" > /dev/null 2>&1; ssh $n "tc qdisc show | grep -c netem" | sed "s/^/reset $n netem qdiscs: /"; done; }
trap reset EXIT
for n in flare polynize; do ssh $n "LATENCY_MS=$LAT BANDWIDTH_MBIT=$BW bash $SH" | tail -1; done
[ -f $OUT ] || echo "pair,name,tag,wall,pre,online,acc,tot,hash,np,phases,live,mb_pre,waited,err,conv_line" > $OUT
for r in $(seq 1 $R); do
  for N in wan_a2b_rca wan_a2bk0_rca wan_rs_rca; do for V in n4096 n8192 rp; do
    ssh flare "pkill -f '^/root/vx/'"; ssh polynize "pkill -f '^/root/vx/'"; sleep 1
    case $V in n4096) B=$N; E="CONV_POLY_N=4096";; n8192) B=$N; E="CONV_POLY_N=8192";; rp) B=${N}_rp; E="";; esac
    T=${N}_${V}_l${LAT}_r$r
    row=$(bash /tmp/mx/vrun.sh fp $B $T "$E")
    cl=$(ssh flare "grep -a 'TRIPLE_STATS (Aggregated)-- CONV' /root/vx/$T.out0 | head -1 | sed -E 's/.*s PRE: ([0-9.]+).*MB SENT PRE: ([0-9.]+).*MB RECEIVED PRE: ([0-9.]+).*/\1;\2;\3/'")
    echo "$row,$V,$cl" >> $OUT; echo "$row,$V,$cl"
  done; done
done
echo WAN2_DONE
