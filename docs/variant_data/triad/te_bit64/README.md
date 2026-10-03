# TE0 / TE1 in 2PC and the 64-bit all_opt builds (2026-10-03, flare / polynize)

Driver scripts (`terun*.sh`, on coinbase), build lists (`list_*.txt`, "NAME FLAGS..."), results (vsum.sh rows; note that
vsum prints no mb_live, so `waited;rounds` sits in that column) and P0's traffic (`comm*.csv`, commcsv.sh, MB sent +
received). `ana.py RES COMM`: medians; `tables.py te|x64 RES COMM`: the paper's table rows.

* `res_acc.csv`: CIFAR-10 accuracy (AdamW ResNet50, 256 images, hpmpc 31c5270): `e_c{1,2,3}[f8]_te{1,0}` TE1 / TE0 at F = 5 / 8
  (158 / 185 in every UC), `e_c2[f8]_tl` TS{L} (159 / 169), `e_c2[f8]_t1` TS1 (161 / 187).
* `res_acc64.csv`: the same at 64 bits, UC2 A2bits RCA, F = 12 / 16 (hpmpc acc5728): TS{L}, TE1, TE0 188 / 189.
* `res3_x64.csv`, `comm3_x64.csv` (hpmpc 4c2e446, `list_x64.txt`): the 18 list_fin2 COMPRESS=0 builds at 32 bits (F = 5)
  and 64 bits (F = 12), and 64-bit UC1 / UC2 A2bits RCA with `CHEETAH_CONV_REPACK=1` (`x64r_*`).
* `res3_im2.csv`, `comm3_im2.csv` (hpmpc 4c2e446, `list_im2.txt`): ImageNet TS{L} (`_xl`), TS1 (`_x1`), TE1, TE0 for
  UC1-3 x RCA / PPA / PPA4, and `_xln` = `CUT_NARROW_32=0` (the old identity-substituted 32-bit PPA / PPA4 cut).
  TE1 there still converted twice; `res4_imte.csv` (ef298d1) converts once; `res5_imte.csv` (8c0ff0f, the final code)
  also builds the low adders on the pool.
* `res6.csv`, `comm6.csv` (hpmpc 8c2adac, `terun6.sh`, `list_all6.txt`): the TE / TS table again (a-known narrow PPA4,
  TE's four-way low adder and B2A products, the folded a-known RCA with the narrow cut), `_xln0` = `CUT_NARROW_32=0` (the
  old identity-substituted RCA), and the nine 64-bit A2bits builds (`x64_*`; a-known PPA4, folded RCA).
* `res7.csv`, `comm7.csv` (hpmpc a058c36, `terun7.sh`, `list_all7.txt`, names `m7_*`): the PPA4 builds with one mask share
  per dot-group root, against `A2BITS_PPA4_AB=1` (`*ab`), 32 and 64 bits.
