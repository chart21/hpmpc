#!/bin/bash
# mxseq.sh PAIR: the whole matrix on one host pair (after the builds finished)
P=$1
A=flare; [ $P = ag ] && A=algofi
until ssh $A "[ \$(grep -c '^ok\|^FAIL' /root/vball.log) -ge 120 ]"; do sleep 10; done
bash /tmp/mx/mxall.sh $P /tmp/mx/list_s_t.txt 2
bash /tmp/mx/mxall.sh $P /tmp/mx/list_m_t.txt 1
bash /tmp/mx/mxall.sh $P /tmp/mx/list_s_r.txt 1
bash /tmp/mx/mxall.sh $P /tmp/mx/list_m_r.txt 1
echo MX_DONE $P
