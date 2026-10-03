#!/bin/bash
# terun.sh (coinbase): hpmpc 31c5270 on flare / polynize; TE accuracy (CIFAR 256), TE ImageNet runtime, 64 vs 32 bits
cd /tmp/te
# (the nodes are at 31c5270 already; rebuild only what is missing)
cat list_acc.txt list_imte.txt list_x64.txt > list_all.txt
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_all.txt $n:/root/list_te.txt && ssh $n "while read x rest; do [ -x /root/vx/\$x.p$p ] || echo \"\$x \$rest\"; done < /root/list_te.txt > /root/list_te_miss.txt; wc -l < /root/list_te_miss.txt; bash /root/vball.sh /root/list_te_miss.txt $p /root/hpmpc/nn/ConvTriple 16 > /root/vb_te.log 2>&1; echo \$(hostname) built \$(grep -c ^ok /root/vb_te.log) failed \$(grep -c FAIL /root/vb_te.log); grep FAIL /root/vb_te.log | head -3" ) &
done; wait
echo BUILT $(date +%T)
MX_MODEL=wd MX_OUT=/tmp/te/res_acc.csv bash /tmp/mx/mxall.sh fp /tmp/te/list_acc.txt 1 > /dev/null 2>&1
echo ACC_DONE $(date +%T)
for L in imte x64; do
  OUT=/tmp/te/res_$L.csv
  echo "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors" > $OUT
  for r in 1 2 3; do for N in $(awk '{print $1}' list_$L.txt); do
    ssh flare "pkill -f '^/root/vx/'"; ssh polynize "pkill -f '^/root/vx/'"; sleep 1
    bash /tmp/mx/vrun.sh fp $N ${N}_r$r "" >> $OUT 2>/dev/null
  done; echo "$L rep $r $(date +%T)"; done
  ssh flare "bash /root/commcsv.sh \$(ls /root/vx/*_r1.out0 | grep -E '/($(awk '{print $1}' list_$L.txt | tr '\n' '|' | sed 's/|$//'))_r1.out0')" > /tmp/te/comm_$L.csv
done
echo TERUN_DONE $(date +%T)
