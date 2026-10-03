#!/bin/bash
# run910.sh (coinbase): after run8 -- run 9: the 18 64-bit all_opt builds (F = 12, LAN, 3 reps); run 10: the 12 UC1 / UC2
# all_opt builds with and without output repacking in the WAN (20 ms / 200 Mbit/s, 2 reps; shaping reset in a trap)
cd /tmp/c8
until grep -q RUN8_DONE run8.log; do sleep 30; done
cat list_c9.txt list_c10.txt > list_c910.txt
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_c910.txt $n:/root/list_c910.txt && ssh $n "bash /root/vball.sh /root/list_c910.txt $p /root/hpmpc/nn/ConvTriple 16 > /root/vb_c910.log 2>&1; echo \$(hostname) built \$(grep -c ^ok /root/vb_c910.log) failed \$(grep -c FAIL /root/vb_c910.log); grep FAIL /root/vb_c910.log | head -5" ) &
done; wait
echo BUILT910 $(date +%T)
OUT=/tmp/c8/res9.csv; head -1 res8.csv > $OUT
for r in 1 2 3; do for N in $(awk "{print \$1}" list_c9.txt); do
  ssh flare "pkill -f ^/root/vx/"; ssh polynize "pkill -f ^/root/vx/"; sleep 1
  bash /tmp/mx/vrun.sh fp $N ${N}_c9r$r "" >> $OUT 2>/dev/null
done; echo "run9 rep $r $(date +%T)"; done
ssh flare "cd /root/vx && bash /root/commcsv.sh \$(for n in \$(cut -d\" \" -f1 /root/list_c910.txt); do [ -f \${n}_c9r1.out0 ] && echo \${n}_c9r1.out0; done)" | sed "s/_c9r1.out0//" > /tmp/c8/comm9.csv
echo RUN9_DONE $(date +%T)
SH=/root/hpmpc/measurements/network_shaping/shape_network_alt.sh
reset() { for n in flare polynize; do ssh $n "LATENCY_MS=-1 BANDWIDTH_MBIT=-1 bash $SH" > /dev/null 2>&1; ssh $n "tc qdisc show | grep -c netem" | sed "s/^/reset $n netem qdiscs: /"; done; }
trap reset EXIT
for n in flare polynize; do ssh $n "LATENCY_MS=20 BANDWIDTH_MBIT=200 bash $SH" | tail -1; done
OUT=/tmp/c8/res10.csv; head -1 res8.csv > $OUT
for r in 1 2; do for N in $(awk "{print \$1}" list_c10.txt); do
  ssh flare "pkill -f ^/root/vx/"; ssh polynize "pkill -f ^/root/vx/"; sleep 1
  bash /tmp/mx/vrun.sh fp $N ${N}_c10r$r "" >> $OUT 2>/dev/null
done; echo "run10 rep $r $(date +%T)"; done
ssh flare "cd /root/vx && bash /root/commcsv.sh \$(for n in \$(cut -d\" \" -f1 /root/list_c10.txt); do [ -f \${n}_c10r1.out0 ] && echo \${n}_c10r1.out0; done)" | sed "s/_c10r1.out0//" > /tmp/c8/comm10.csv
echo RUN10_DONE $(date +%T)
