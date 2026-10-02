# Round 8 (the bake's Boolean addition), flare / polynize, 2026-10-02

hpmpc wip-adder 79de8cf..a56585d (merged into dbg as ddc082f), ConvTriple 5c8adc1..70f3eab. ImageNet dummy (-1/-1),
fin9 flags (list_ad5.txt: `_off` = A2B_ADDER_BATCH=0 A2B_ADDER_CUT=0, `_on` = the new defaults).

* res_ad5.csv: the 18 A2bits builds off / on, 2 interleaved runs (columns name, env, pre, online, err, Boolean addition
  s, its log line: total, random OTs, first-round wait). The main result.
* res_ad1.csv (b0 = before, b1 = batch, b2 = batch+cut, but the cut was not active yet: include order), res_ad2.csv (b1 /
  b2 with the cut active), res_ad3.csv (A2B_ROUND_CHANNELS 4 / 1 / 32: no difference), res_t2.csv (contiguous loops).
* res_adc*.csv: CIFAR-10 AdamW, 10 images, single batch (b0 / b2); res_pm5.csv: multi-batch 24 x 8 images,
  MODELWEIGHTS_KNOWN 0 / 1 (134 / 124 of 192).
