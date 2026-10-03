# CHEETAH_CONV_POLY_N: N = 4096 vs 8192 for the 32-bit packed convs (2026-10-03, flare / polynize)

hpmpc ecdb9cf, ConvTriple f964c7c. ImageNet ResNet50, dummy weights, the list_fin2 builds `a2b_rca_c0` (UC2 A2bits),
`a2bk0_rca_c0` (UC1 A2bits, AB conv triples) and `rs_rca_c0` (UC2 reshared) as `wan_*` (`list_wan.txt`); N set at run
time (`CONV_POLY_N`). `wan.sh`: LAN once (`lan.csv`), then WAN with `shape_network_alt.sh` (`LATENCY_MS=20
BANDWIDTH_MBIT=200` on both hosts, all interfaces; reset with -1 / -1 on exit) 3 times interleaved (`wan.csv`;
vsum.sh columns, plus the CONV triple line: s, MB sent / received by P0). Medians:

| build | N | LAN pre s | WAN pre s | WAN online s | WAN conv s | conv MiB (P0 sent + recv) |
|---|---|---|---|---|---|---|
| UC2 A2bits RCA | 4096 | 3.17 | 30.39 | 32.12 | 16.87 | 479.1 |
| | 8192 | 3.29 | 33.52 | 32.19 | 20.91 | 628.7 |
| UC1 A2bits RCA | 4096 | 3.20 | 36.98 | 32.30 | 26.74 | 958.2 |
| | 8192 | 3.28 | 43.33 | 32.40 | 33.25 | 1257.5 |
| UC2 reshared RCA | 4096 | 2.05 | 25.02 | 33.54 | 15.79 | 479.1 |
| | 8192 | 2.12 | 28.99 | 33.52 | 19.53 | 628.7 |

N = 8192 sends 1.31x the conv-triple traffic (the sparse output layout costs ~sqrt(N) per output element) and is
slower in both settings; in the WAN the conv phase is bandwidth-bound (1.24x the time). The online phase does not
depend on N (in the WAN it is round-bound: 1714-1818 rounds at 20 ms).
