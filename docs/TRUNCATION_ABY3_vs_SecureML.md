# Truncation in HPMPC ABY2 2PC (PROTOCOL=4): when SecureML vs ABY3 works

Fixed-point multiply produces a value scaled by `2^(2f)`; we must truncate by `f` bits to return to
`2^f` scale. This note summarizes which truncation method is correct in which case, based on the ABY2
2PC sharing used here.

## Sharing model
A value `v` is held as `(m, λ)` where **`m` is public** (identical on both parties after the
multiply's reveal) and **`λ = λ_0 + λ_1` is secret-shared**; `v = m − λ`. The product, *before*
truncation, is available either as an **additive sharing** `m_0 + m_1 = v·2^f` (each party holds one
summand) or as a **masked value** `m = v·2^f + λ`.

Ring size `2^k` (`k = BITLENGTH = 32`), fractional bits `f = FRACTIONAL`.

---

## 1. SecureML local truncation (current default, `TRUNC_APPROACH=0`)
**Mechanism:** each party locally right-shifts its own additive share: `y_i = x_i >> f`. Sum
`y_0 + y_1 = trunc(x) + e`. (In code: `mask_and_send_dot_with_trunc`, with the `-TRUNC(-m)` trick on P0.)

**Correct when:** the *value* `x = v·2^f` is **small relative to the ring** (`ℓ_x ≪ k`). The large
"wrap" error `2^(k−f)` only occurs when `x_0 + x_1` overflows `2^k`; for an additive sharing of a
bounded `x` this happens with probability `≈ 2^(ℓ_x + 1 − k)`, which is negligible for NN activations
in a 32-bit ring. A residual `±1`-ULP rounding error remains (probabilistic, harmless).

**Fails when:**
- `x` is large (close to the ring): the wrap fires → output off by `2^(k−f)` (`−2^22` in float here).
- The `-TRUNC(-m)` formula with a **logical** shift adds a *structural* `−2^(k−f)` constant unless the
  triple's `λ` compensates. The **a-known / AB2 triple** (`A_KNOWN=1`) supplies that compensation in
  its PRE-phase `λ`; the **regular AB triple** (`A_KNOWN=0`) historically did not.

**Net:** SecureML is fine for conv/FC on **bounded** activations and is what makes `A_KNOWN=1` work.
It is single-round (truncation is local, folded into the multiply's reveal).

---

## 2. ABY3 masked-reveal / truncation-pair (`ABY3_PROB_TRUNC`, experimental)
**Mechanism:** reveal the masked value `m_c = v·2^f ± r` for a mask `r`, truncate `m_c` **publicly**
(exact, both parties agree), then add a precomputed share `[trunc(r)]`:
`trunc(v) = trunc(m_c) + [trunc(r)]`.

This is correct **iff two conditions hold**:
1. **`[trunc(r)]` is exact** — a true share of `trunc(r)`, *not* a sum of locally-truncated shares.
2. The reveal `m_c = v·2^f − r` does **not** wrap, *or* the wrap is corrected (an MSB term).

**Where `[trunc(r)]` can be made exact:**
- **A helper/dealer holds the full `r`** → truncates locally, shares the result. This is the 3PC/4PC
  case (`oecl`, `oec-mal`) and also the ABY2 **a-known** case where one party effectively holds the
  full mask. ⇒ this is *why `A_KNOWN=1` BatchNorm/conv are correct*: the model-weight mask is held by
  one party, so its truncation is exact.
- **edaBits / A2B + bit extraction** of a shared `r` (general 2PC, expensive).
- **Exact `trunc_2k`** additionally precomputes `msb(r)` and corrects the reveal wrap with
  `(msb(m_c) XOR msb(r))·2^(k−f−1)` ⇒ always correct (1-ULP). This is condition (2) done properly.

**Where it FAILS (measured here):**
- `[trunc(r)]` from **summing local truncations** of `r`'s shares → reintroduces the `2^(k−f)` wrap of
  `r_0 + r_1`. No good.
- The **"pre-truncate the mask factors by f/2"** shortcut (`trunc_{f/2}(λx)·trunc_{f/2}(λw) ≈
  trunc_f(λx·λw)`): only valid for **bounded** masks. With **full-range** masks `λ ~ 2^k`, the cross
  term `ε·λw / 2^(f/2)` is up to `2^(k−f/2)` ⇒ `[trunc(r)]` is off by `~2^(k−f/2)`. **Confirmed
  experimentally:** the conv unit test (func 53) with `ABY3_PROB_TRUNC=1, A_KNOWN=0` outputs garbage of
  magnitude `~2^25` float (`= 2^(k−f/2)/2^f`), matching this term exactly. An arithmetic (sign-aware)
  shift on `m_c` does **not** fix it, because the error is inside `[trunc(r)]`, not the reveal.
- If `r` (the mask used in `m_c = v·2^f − r`) is full-range, `m_c` wraps almost always; without the
  `msb` correction this leaves a `2^(k−f)` term even if `[trunc(r)]` were exact.

---

## Summary table

| Case | Truncation that works | Why |
|---|---|---|
| `A_KNOWN=1` conv/FC (model-weight mask held by 1 party) | a-known triple (ABY3-exact `[trunc(λ)]`) | one party truncates the full mask exactly |
| `A_KNOWN=1` BatchNorm | same | scale mask held by model owner |
| `A_KNOWN=0` conv/FC, **small** activations | SecureML local | wrap prob `2^(ℓ_x+1−k)` negligible |
| `A_KNOWN=0`, **large/both-shared** values | exact `trunc_2k` (reveal + `[trunc(r)]` + `msb` correction) | only method that removes the wrap with a split mask |
| `A_KNOWN=0` with f/2-pretruncated **full-range** masks | **none** — broken | `2^(k−f/2)` cross-error |
| `A_KNOWN=0` with **bounded** masks (slack) | ABY3 masked-reveal (probabilistic) | small mask ⇒ no `m_c` wrap, small cross-error |

## Recommendation for `A_KNOWN=0` / ResNet BatchNorm
The masked-reveal (ABY3) family is the right direction, but in 2PC with a **split, full-range** mask it
needs **either**:
- **(a)** the exact `trunc_2k` MSB correction (precompute `msb(r)` alongside `[trunc(r)]`; `[trunc(r)]`
  itself from the dealer/HE that already produces the conv/FC/BN triple, truncating the **product**
  `λx·λw` as a whole — not each factor by f/2), **or**
- **(b)** bounded masks (reduced-entropy slack), which avoids both the `m_c` wrap and the cross-error
  but weakens statistical hiding.

The `f/2`-per-factor shortcut cannot work with full-range masks and should not be pursued further.

---
*Status: the `ABY3_PROB_TRUNC` experiment is **not part of this branch**. It lives on a separate branch,
where it is not correct for full-range masks (the conv unit test fails with it on). This branch uses the
SecureML truncation throughout, plus the a-known variant for the first layer under `PUBLIC_WEIGHTS`.*
