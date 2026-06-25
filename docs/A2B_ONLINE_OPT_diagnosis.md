# A2B_ONLINE_OPT — diagnosis, root cause, and fix design

**Status:** root cause **definitively found**; fix design **confirmed**; implementation **infrastructure started, behind `A2B_ONLINE_OPT==1`.** Protocol = 4 (Cheetah 2PC ABY2), `aby2-merge` branch.

This document is the durable record of the A2B_ONLINE_OPT investigation. (The separately-validated **RESHARE_OPT** fix — a use-before-def statement-ordering bug in `ppa_msb_unsafe_and_ab_reshared.hpp`, giving LeNet 100% / ResNet50 70% — is already in `HEAD` and is summarized at the end.)

## TL;DR

`A2B_ONLINE_OPT=1` produces wrong ReLU/MaxPool signs (func53: 4/6, networks broken). The cause is **not** the boolean addition (which is correct). It is the **MSB adder re-masking its S2 operand using a PRE value that differs from the online value.**

## What A2B_ONLINE_OPT does

The A2B (arithmetic→boolean) for the MSB/sign uses a boolean adder over two operands:
- **S1 = bits(m)** — the masked value `m` is public, so S1 is public (`l=0` on both parties). This is the optimization (no online sharing of S1).
- **S2 = bits(-λ)** where `λ = l_0 + l_1` is the secret mask. `-λ` is precomputed in the preprocessing by a **2-party boolean ripple-carry addition** of `bits(-l_0)` (P0) and `bits(-l_1)` (P1), producing XOR-shares `c`. Online, `S2.l = c`.

The MSB adder computes `MSB(S1 + S2) = MSB(m - λ) = sign(v)`.

## Proven correct (ruled out as the cause)

1. **The boolean addition is correct.** Isolation test (`test_bool_addition_isolated` in `core/generate_beaver_tiples.hpp`, run under `DEBUG_A2B=1`) feeds known `l_0,l_1` and checks `c_P0 ⊕ c_P1 == bits(-(l_0+l_1))`. Passes for **all** cases: small, large, MSB-carry-out, alternating bits, and all 32 SIMD lanes filled with distinct values (multi-element). The PRE ripple-carry adder (`generateBooleanAdditionDummyTriples`) is sound.
2. **The `c → online S2` consumption is correct.** A "tag" test (overwrite `c` with the element index) + a buffer-pointer trace showed online `S2.l` reads exactly the addition result from the right buffer slot.
3. **`S2 = bits(-λ)` exactly.** Reconstructing the online `S2.l` (both parties print, XOR offline) yields exactly `-λ = v - m` (verified `-609238991`, `-543993127`, `-1665930140`, …).
4. **`S1 = bits(m)` exactly** (`DEBUG_A2B` reveal == the public `m` read directly via `get_m_debug()`).

> **Probe lesson:** the `DEBUG_A2B` *reveal* of S2 is **unreliable** for A2B_ONLINE_OPT=1, because the **PRE** `prepare_A2B_S2` stores `S2.l = bits(-l_i)` (the input) while **online** stores `S2.l = c` (the result); the reveal mixes them. Use the online-`S2.l` reconstruction or the MSB-output reveal instead. The MSB-output reveal is reliable (msb's PRE share == online share).

## Root cause

A reliable reveal of the **adder's MSB output** vs `sign(v)` shows the MSB is **wrong** (e.g. `v=3 → msb=1`; carry-into-MSB computed `0` instead of `1`). Since S1 and S2 are both correct, the bug is in the **adder**.

The MSB adder re-masks each operand with a Beaver-triple mask via **`zero_add`** (or `reshare_b` under `RESHARE_OPT`). `zero_add` only preserves the operand value if **`PRE operand.l == online operand.l`**. For S2 (and every wire derived from it):
- **online** `S2.l = c = bits(-λ)`
- **PRE** `S2.l = bits(-l_i)` (set in `aby2_pre.hpp` `prepare_A2B_S2`, line ~568)

Because `bits(-l_0) ⊕ bits(-l_1) ≠ bits(-λ)` (XOR ≠ ADD), the `zero_add` pre-send disagrees PRE vs online → the re-masked operand's public `m` part differs between the two parties (no longer public) → the Beaver AND is corrupted → wrong carry → wrong MSB. Independent of the (correct) boolean addition.

Confirmed: neither `A_KNOWN_TO_EVALUATORS_OPT=1` (the `a_ab` adder) nor `RESHARE_OPT=1` fixes it — both still re-mask S2 with the inconsistent PRE `l`.

## Why it can't be patched locally

`c` is produced by the boolean addition **during triple generation**, which runs **after** the PRE forward-pass dry run that already executed the adder with `bits(-l_i)`. The adder cannot see `c` at that point.

## Fix design (chosen with the user — option A, two-pass / deferred)

During the PRE forward pass, **skip all boolean circuits** (the MSB adders, and — because the deferral cascades, see below — the bit-injections / comparisons that consume the adder's `msb`). Store the `-lv` inputs (already collected into `boolean_addition_triple_a/b`). Then, **after** the boolean addition produces `c`, run all deferred boolean circuits in **one batched loop** with `S2.l = c`. Nothing is recomputed.

**Cascade (important scope note):** deferring only the adder leaves the PRE `msb` a placeholder while online `msb` is real — the *same* PRE≠online inconsistency then breaks every downstream boolean op that re-masks `msb` (e.g. ReLU bit-injection). So the deferral must cover **all** boolean circuits dependent on the A2B output — a PRE-phase restructure, not a localized adder change.

**Dedicated buffer (chosen):** the deferred adders' `zero_add` output-shares must **not** use the shared default buffer `preprocessed_outputs` — deferring them there misaligns with the online's mid-network reads. They get a **dedicated buffer** (`preprocessed_outputs_a2b`) filled in forward-pass order, so PRE/online stay aligned.

**De-risking facts established:**
- The **only** output-share the adder itself reads online is `zero_add`'s `retrieve_output_share()` (default buffer) → that is the single read to route to the dedicated buffer. `prepare_mult` uses the passed Beaver-triple `c` (not an output-share read).
- `pre_send_to_live`/`pre_receive_from_live` are per-player FIFOs; in the PRE adder, **only `zero_add` pre-sends** (`prepare_mult` for `ROT_PREPROCESSING_OPT=1` just pushes `CaseAND` and returns a random mask).
- `CaseAND` (`receive_and_compute_lxly_share`) merely copies `boolean_triple_c` into `lxly_b`; the Beaver-triple side stays aligned as long as adders run in forward-pass order.

## Implementation state (this checkpoint)

All gated on `#if A2B_ONLINE_OPT == 1` (baseline `A2B_ONLINE_OPT=0` unaffected; baseline func53 still 6/6):
- `core/networking/buffers.h`: dedicated-buffer globals `preprocessed_outputs_a2b` + indices + `g_a2b_adder_active` flag.

Debug/diagnostic instrumentation, gated on `#if DEBUG_A2B == 1` (default 0):
- `config.h` / `Makefile`: the `DEBUG_A2B` flag.
- `programs/functions/share_conversion.hpp`: per-element reveals of S1/S2/m/v and the reliable MSB-output reveal in `get_msb_range`.
- `core/generate_beaver_tiples.hpp`: `test_bool_addition_isolated` (known-value isolation test of the boolean addition) + per-bit carry capture + (disabled) tag-overwrite trace.
- `protocols/2-PC/aby2/aby2_{online,pre,init}.hpp`: `get_m_debug()` accessors + consumption/buffer trace prints.

Unit test: `programs/tests/test_conv_pool.hpp` `relu_large_test` (`TEST_RELU_LARGE`) — ReLU on values up to ±7.5 (exercises the integer carry chain that the tiny `<1` inputs hide). Drives the func53 (FUNCTION_IDENTIFIER 53) test harness.

Debug helpers: `scratch/adder_debug/find_ubd.py` (flags use-before-def of prefix wires in a generated adder — this is what found the RESHARE_OPT bug) and `verify_adder.py` (Beaver-mask telescoping check).

## Remaining work

1. Defer every boolean circuit in `get_msb_range` (and the bit-injection path) during the PRE forward pass; store `(s1, s2, msb, len)`.
2. Batched run after the `BOOLEANADDITION` generation in `complete_preprocessing`: set `S2.l = c` (needs a setter into the private share `l`), run the deferred circuits with `g_a2b_adder_active=true`.
3. Route `zero_add`'s retrieve/store to `preprocessed_outputs_a2b` under the flag (PRE store + online read), with its own send/receive in the batch.
4. Repeat for the bit-injection / downstream boolean circuits.

## Appendix — RESHARE_OPT fix (already in HEAD)

`RESHARE_OPT=1` with the default `PPA_MSB` adder gave LeNet 10% / ResNet50 10–20%. Root cause: a **use-before-def statement-ordering bug** in the auto-generated `programs/functions/adders/zero_add_adders/ppa_msb_unsafe_and_ab_reshared.hpp` — `g_L3_X = g_L2_X ^ pg_L2_X_Y;` was emitted *before* the `g_L1_X`/`g_L2_X` chain that produces `g_L2_X`, so the level-3 prefix-generate consumed an uninitialized `g_L2_X`. Input-dependent (only wrong when the carry propagates through that node), so small `<1` values passed and ±6 values failed. Fixed 4 sites (k16: `g_L3_1`; k32: `g_L3_8/24/16`) by reordering to `g_L1→g_L2→g_L3(→g_L4)` (all local XORs, safe). Results: func53 6/6, **LeNet 182 = 100%** (was 10%), **ResNet50 171 = 70%** = baseline (was 10–20%).
