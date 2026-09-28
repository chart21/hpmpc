#!/bin/bash
# vb.sh NAME PARTY CT FLAGS...: compile one hpmpc variant (run-P<PARTY>) into /root/vx/NAME.p<PARTY>
N=$1 P=$2 C=$3; shift 3
cd /root/hpmpc && mkdir -p /root/vx
D=""; for f in "$@"; do D="$D -D$f"; done
T0=$(date +%s)
if g++ main.cpp -include core/include/pch.h -march=native -Ofast -fno-finite-math-only -std=c++20 -pthread -Wno-ignored-attributes \
  -I$C/src/include -I$C/src -isystem $C/deps/include -isystem $C/deps/include/SEAL-4.1 -isystem /usr/include/eigen3 \
  -Wl,-rpath,$C/build/lib:$C/deps/lib -L$C/build/lib -L$C/deps/lib -lHE -lgemini -lseal-4.1 -lssl -lcrypto -I nn/PIGEON \
  $D -DPARTY=$P -DSPLIT_ROLES_OFFSET=0 -o /root/vx/$N.p$P > /root/vx/$N.p$P.log 2>&1; then
  echo "ok $N $(( $(date +%s) - T0 ))s"
else
  rm -f /root/vx/$N.p$P; echo "FAIL $N: $(grep -m1 ' error' /root/vx/$N.p$P.log | cut -c1-200)"
fi
