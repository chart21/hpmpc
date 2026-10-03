#!/bin/bash
# terun3.sh (coinbase): hpmpc 4c2e446 (PIGEON 77b0644): TE ImageNet (pooling folds), 64-bit runtime and
# accuracy, CUT_NARROW_32=0 references; every build of the lists rebuilt on the new code
cd /tmp/te

for n in flare polynize; do (scp -q /tmp/nu3.sh /tmp/te/vb.sh $n:/root/ && ssh $n "bash /root/nu3.sh 4c2e446 2d0b448 77b0644" 2>&1 | grep -v Warning | sed "s/^/$n: /") & done; wait
cat list_im2.txt list_x64.txt > list_all2.txt
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_all2.txt $n:/root/list_te3.txt && ssh $n "bash /root/vball.sh /root/list_te3.txt $p /root/hpmpc/nn/ConvTriple 16 > /root/vb_te3.log 2>&1; echo \$(hostname) built \$(grep -c ^ok /root/vb_te3.log) failed \$(grep -c FAIL /root/vb_te3.log); grep FAIL /root/vb_te3.log | head -3" ) &
done; wait
echo BUILT3 $(date +%T)


for L in im2 x64; do
  OUT=/tmp/te/res3_$L.csv
  echo "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors" > $OUT
  for r in 1 2 3; do for N in $(awk '{print $1}' list_$L.txt); do
    ssh flare "pkill -f '^/root/vx/'"; ssh polynize "pkill -f '^/root/vx/'"; sleep 1
    bash /tmp/mx/vrun.sh fp $N ${N}_w$r "" >> $OUT 2>/dev/null
  done; echo "$L rep $r $(date +%T)"; done
  ssh flare "cd /root/vx && bash /root/commcsv.sh \$(for n in \$(cut -d' ' -f1 /root/list_te3.txt); do [ -f \${n}_w1.out0 ] && echo \${n}_w1.out0; done)" | sed 's/_w1.out0//' > /tmp/te/comm3_$L.csv
done
echo TERUN3_DONE $(date +%T)
