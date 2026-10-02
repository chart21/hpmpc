# TS1 / TS_Mix in 2PC (2026-10-02, flare / polynize)

`res_im.csv` (vrun.sh rows: pre s, online s, ..., `waited;rounds`), `comm_im.csv` (commcsv.sh, MB sent + received by
P0), `table.txt` (`ana.py res_im.csv comm_im.csv`, medians, MiB). Builds: `list_r6.txt` = the round-8 `_c0_on`
builds (`list_im_base.txt`, `_tl` = TS{L} as given) plus `TRUNC_APPROACH=4 TRUNC_DELAYED=1` (`_tf`), and UC3's
TS{L} with the ResNet TD=1 downsample fix (`_tn`). ImageNet: dummy weights. `res_r6.csv` also has the CIFAR runs
(AdamW model, 100 images): `cif_c1h` / `cif_c2h` / `cif_c3n` TS{L}, `tf_c*h` TS_Mix, `tf_c2wh` with TS1_LOW_CARRY.

## Final round (shifted cut design, pooling fold, A2B_DELAYED_CUT; hpmpc after f10eb8c)

* `list_im_final.txt`: ImageNet builds (dummy weights) per UC (`a2bk0` UC1, `a2b` UC2, `a2bpw` UC3) and adder:
  `_xl` TS{L} as given (UC3: TD=1, A2B_DELAYED_CUT default on), `_xl0` UC3 TS{L} with `A2B_DELAYED_CUT=0`, `_xm` TS_Mix
  (`TRUNC_APPROACH=4 TRUNC_DELAYED=1`, shifted design), `_xw` TS_Mix with `TS1_LOW_CARRY=1` (full design), `_x1` TS1
  (`TRUNC_APPROACH=1`). `res_im_final.csv` (3 interleaved runs), `comm_im_final.csv` (commcsv.sh, MB),
  `table_final.txt` (`ana_final.py comm_im_final.csv res_im_final.csv`, medians, MiB).
* `res_online.csv`: the `_xl/_xm/_x1` builds again after the bit-injection change (M, sK recomputed from M0), 4 runs.
* `list_f.txt`, `res_f.csv`, `comm_f.csv`: RCA builds with `FRACTIONAL=8` (TS{L}, TS_Mix) and `=10` (TS_Mix),
  interleaved with the F = 5 `_xl` / `_xm` builds, 3 runs.
* `list_acc256.txt`, `res_acc256.csv`, `acc_table.md` (`acc_tab.py`): CIFAR-10 accuracy, AdamW ResNet50
  (`MODEL_DIR=nn/Pygeon/models/pretrained/adam_001_wd MODEL_FILE=ResNet50_avg_AdamW_d05_wd003_lr0001_ep100_acc74_35.bin`),
  256 images, `a_c{1,2,3}[f8|f10]_{tl,tl0,mx,mxw,t1,t1w}` (`mx` TS_Mix, `t1` TS1, `w` TS1_LOW_CARRY). Column 7: correct.
