#!/bin/bash
# run8.sh (coinbase): hpmpc 4b43984 (ConvTriple 82b1400, PIGEON 77b0644): the all_opt campaign from gen_configs.py --
# the 36 all_opt builds, the truncation variants (A2bits), the cut variants (none / identity / narrow); 3 interleaved reps
cd /tmp/c8
for n in flare polynize; do (scp -q /tmp/nu3.sh /tmp/te/vb.sh $n:/root/ && ssh $n "bash /root/nu3.sh 4b43984 82b1400 77b0644" 2>&1 | grep -v Warning | sed "s/^/$n: /") & done; wait
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_c8.txt $n:/root/list_c8.txt && ssh $n "bash /root/vball.sh /root/list_c8.txt $p /root/hpmpc/nn/ConvTriple 16 > /root/vb_c8.log 2>&1; echo \$(hostname) built \$(grep -c ^ok /root/vb_c8.log) failed \$(grep -c FAIL /root/vb_c8.log); grep FAIL /root/vb_c8.log | head -5" ) &
done; wait
echo BUILT8 $(date +%T)
OUT=/tmp/c8/res8.csv
echo "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors" > $OUT
for r in 1 2 3; do for N in $(awk "{print \$1}" list_c8.txt); do
  ssh flare "pkill -f ^/root/vx/"; ssh polynize "pkill -f ^/root/vx/"; sleep 1
  bash /tmp/mx/vrun.sh fp $N ${N}_c8r$r "" >> $OUT 2>/dev/null
done; echo "rep $r $(date +%T)"; done
ssh flare "cd /root/vx && bash /root/commcsv.sh \$(for n in \$(cut -d\" \" -f1 /root/list_c8.txt); do [ -f \${n}_c8r1.out0 ] && echo \${n}_c8r1.out0; done)" | sed "s/_c8r1.out0//" > /tmp/c8/comm8.csv
echo RUN8_DONE $(date +%T)
