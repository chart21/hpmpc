#!/bin/bash
# terun2.sh (coinbase): after terun.sh; hpmpc acc5728 (PIGEON 77b0644): TE ImageNet (pooling folds), 64-bit runtime and
# accuracy, CUT_NARROW_32=0 references; every build of the lists rebuilt on the new code
cd /tmp/te
until grep -q TERUN_DONE terun.log; do sleep 30; done
for n in flare polynize; do (scp -q /tmp/nu3.sh /tmp/te/vb.sh $n:/root/ && ssh $n "bash /root/nu3.sh acc5728 2d0b448 77b0644" 2>&1 | grep -v Warning | sed "s/^/$n: /") & done; wait
cat list_acc64.txt list_im2.txt list_x64.txt > list_all2.txt
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_all2.txt $n:/root/list_te2.txt && ssh $n "bash /root/vball.sh /root/list_te2.txt $p /root/hpmpc/nn/ConvTriple 16 > /root/vb_te2.log 2>&1; echo \$(hostname) built \$(grep -c ^ok /root/vb_te2.log) failed \$(grep -c FAIL /root/vb_te2.log); grep FAIL /root/vb_te2.log | head -3" ) &
done; wait
echo BUILT2 $(date +%T)
MX_MODEL=wd MX_OUT=/tmp/te/res_acc64.csv bash /tmp/mx/mxall.sh fp /tmp/te/list_acc64.txt 1 > /dev/null 2>&1
echo ACC64_DONE $(date +%T)
for L in im2 x64; do
  OUT=/tmp/te/res2_$L.csv
  echo "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors" > $OUT
  for r in 1 2 3; do for N in $(awk '{print $1}' list_$L.txt); do
    ssh flare "pkill -f '^/root/vx/'"; ssh polynize "pkill -f '^/root/vx/'"; sleep 1
    bash /tmp/mx/vrun.sh fp $N ${N}_q$r "" >> $OUT 2>/dev/null
  done; echo "$L rep $r $(date +%T)"; done
  ssh flare "cd /root/vx && bash /root/commcsv.sh \$(for n in \$(cut -d' ' -f1 /root/list_te2.txt); do [ -f \${n}_q1.out0 ] && echo \${n}_q1.out0; done)" | sed 's/_q1.out0//' > /tmp/te/comm2_$L.csv
done
echo TERUN2_DONE $(date +%T)
