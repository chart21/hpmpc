#!/bin/bash
# run.sh (coinbase): build the _rp variants, then LAN once and WAN 3x
cd /tmp/wan2
for h in flare:0 polynize:1; do n=${h%:*}; p=${h#*:}
  ( scp -q list_wan2.txt $n:/root/ && ssh $n "bash /root/vball.sh /root/list_wan2.txt $p /root/hpmpc/nn/ConvTriple 6 | sort | uniq -c | head" ) &
done; wait
bash wan2.sh 1 lan.csv -1 -1 > lan.log 2>&1
bash wan2.sh 3 wan.csv 20 200 > wan.log 2>&1
echo RUN_DONE
