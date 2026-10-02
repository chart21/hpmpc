# All 36 all_opt configs with TS1 / TS_Mix (2026-10-03, flare / polynize, ImageNet, dummy weights)

* `list_ao.txt`: the 36 configs of `../../list_fin2.txt` (`ao_*`, the final code: hpmpc 468167d + the reshared delayed
  cut) plus TS_Mix (`_mx`, `TRUNC_APPROACH=4 TRUNC_DELAYED=1`) and TS1 (`_t1`, `TRUNC_APPROACH=1`) of the nine A2bits
  COMPRESS=0 configs (the only ones with the A2B bake TS1 needs). `res_ao.csv` (3 interleaved runs, before the
  reshared delayed cut), `comm_ao.csv` (commcsv.sh, MB).
* `list_rd.txt`: the reshared UC3 delayed cut (`ao_rspw_*_c0_dc`), its unit tests (`u_rpw_*`, func 59 = RCA only) and
  CIFAR accuracy builds (`a_rpw_*_dc` / `_nd`, AdamW model, 256 images: `res_rd_acc.csv`), and TS1 with
  `FRACTIONAL=8` (`_t1f8`). `list_b2.txt` / `res_b2.csv` / `comm_b2.csv`: these interleaved with their references,
  3 runs.
* `list_pc.txt` / `list_pc2.txt` / `res_pc.csv` / `comm_pc.csv`: PPA4 with and without the cut (UC2 reshared
  `CUT_FRACTIONAL_BITS_OPT=0`, UC3 A2bits `A2B_DELAYED_CUT=0`), 5 runs.
* `ao_table.md`: `python3 ao_final.py .` (medians; `*` = least traffic of the config).
