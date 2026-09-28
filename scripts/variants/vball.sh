#!/bin/bash
# vball.sh LIST PARTY CT JOBS: build every variant of LIST ("NAME FLAGS..." lines) for this node's party
L=$1 P=$2 C=$3 J=${4:-16}
cd /root/hpmpc && [ -f core/include/pch.gch ] || g++ -march=native -Ofast -fno-finite-math-only -std=c++20 -pthread -Wno-ignored-attributes -x c++-header core/include/pch.h -o core/include/pch.gch
grep -v '^#' $L | grep . | xargs -P $J -L 1 bash -c 'bash /root/vb.sh "$0" '"$P $C"' "$@"'
