# TS1 / TS_Mix in 2PC (2026-10-02, flare / polynize)

`res_im.csv` (vrun.sh rows: pre s, online s, ..., `waited;rounds`), `comm_im.csv` (commcsv.sh, MB sent + received by
P0), `table.txt` (`ana.py res_im.csv comm_im.csv`, medians, MiB). Builds: `list_r6.txt` = the round-8 `_c0_on`
builds (`list_im_base.txt`, `_tl` = TS{L} as given) plus `TRUNC_APPROACH=4 TRUNC_DELAYED=1` (`_tf`), and UC3's
TS{L} with the ResNet TD=1 downsample fix (`_tn`). ImageNet: dummy weights. `res_r6.csv` also has the CIFAR runs
(AdamW model, 100 images): `cif_c1h` / `cif_c2h` / `cif_c3n` TS{L}, `tf_c*h` TS_Mix, `tf_c2wh` with TS1_LOW_CARRY.
