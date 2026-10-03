#!/bin/bash
# terun6.sh (coinbase): hpmpc 8c2adac (ConvTriple 82b1400, PIGEON 77b0644): the TE / TS table (a-known narrow PPA4, TE
# four-way low adder and B2A products, folded a-known RCA with the narrow cut), old-RCA references (CUT_NARROW_32=0),
# the 64-bit A2bits builds (a-known PPA4, folded RCA); 3 interleaved reps
cd /tmp/te
for n in flare polynize; do (scp -q /tmp/nu3.sh /tmp/te/vb.sh $n:/root/ && ssh $n "bash /root/nu3.sh 8c2adac 82b1400 77b0644" 2>&1 | grep -v Warning | sed "s/^/$n: /") & done; wait
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_all6.txt $n:/root/list_te6.txt && ssh $n "bash /root/vball.sh /root/list_te6.txt $p /root/hpmpc/nn/ConvTriple 16 > /root/vb_te6.log 2>&1; echo \$(hostname) built \$(grep -c ^ok /root/vb_te6.log) failed \$(grep -c FAIL /root/vb_te6.log); grep FAIL /root/vb_te6.log | head -3" ) &
done; wait
echo BUILT6 $(date +%T)
OUT=/tmp/te/res6.csv
echo "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors" > $OUT
for r in 1 2 3; do for N in $(awk "{print \$1}" list_all6.txt); do
  ssh flare "pkill -f ^/root/vx/"; ssh polynize "pkill -f ^/root/vx/"; sleep 1
  bash /tmp/mx/vrun.sh fp $N ${N}_v$r "" >> $OUT 2>/dev/null
done; echo "rep $r $(date +%T)"; done
ssh flare "cd /root/vx && bash /root/commcsv.sh \$(for n in \$(cut -d\" \" -f1 /root/list_te6.txt); do [ -f \${n}_v1.out0 ] && echo \${n}_v1.out0; done)" | sed "s/_v1.out0//" > /tmp/te/comm6.csv
echo TERUN6_DONE $(date +%T)
