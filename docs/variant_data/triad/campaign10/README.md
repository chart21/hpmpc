# The round-9 campaign (2026-10-03, flare / polynize)

hpmpc `4b43984`, ConvTriple `82b1400`, PIGEON `77b0644`; every build from
`measurements/configs/artifacts/2pc_optimizations/gen_configs.py` (plus `PRINT_OUTPUT_HASH=1 NET_WAIT_STATS=1`).

* `run8.sh`, `list_c8.txt`, `res8.csv`, `comm8.csv`: 100 ImageNet builds, three interleaved LAN runs each: the 36
  all_opt builds (`{a2,rs}{1,2,3}_{rca,ppa,ppa4}_c{0,1}`: A2bits / reshared, UC1-3, COMPRESS), the truncation schemes of
  the A2bits COMPRESS=0 builds (`_mix`, `_ts1`, `_te1`, `_te0`: TRUNC_APPROACH 4, 1, 3, 2 with TRUNC_DELAYED=1), the cut
  variants (`_noc`: CUT_FRACTIONAL_BITS_OPT=0; `_idc`: CUT_NARROW_32=0, A2bits only), and `ref_it_a2b_rca_xl` (the
  earlier lists' flags: the same traffic as `a22_rca_c0`).
* `run910.sh`, `list_c9.txt`, `res9.csv`, `comm9.csv`: the 18 64-bit all_opt builds (F = 12, COMPRESS=0), LAN, three
  runs (`x{a2,rs}{1,2,3}_*`).
* `list_c10.txt`, `res10.csv`, `comm10.csv`: the 12 UC1 / UC2 COMPRESS=0 builds without (`_n`) and with (`_rp`) output
  repacking, WAN (20 ms per direction, 200 Mbit/s on both hosts, reset in a trap), two runs.
* `ana10.py trunc|cut|final|x64|wan`: the paper's rows (medians); `final` writes `../res_fp_final10.csv` and
  `../comm_fp10.csv` for `docs/paper/make_triad.py`. `fill_paper.py`: the rows and ranges into the paper.

Units: ConvTriple's counters (HE, OT, key exchange) are MiB, hpmpc's (`hpmpc_pre`, `online`) 10^6 bytes; `ana10.py`
converts the latter. (`te_bit64/tables.py` divided both by 1.048576, which put the HE / OT part of the earlier
truncation and 64-bit tables 4.6% low; the paper's tables now come from here.)
