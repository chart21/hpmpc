#!/usr/bin/env python3
"""he_model.py: traffic of the packed AB2 convolution triples (ConvTriple PackedConv2D) for the CHEETAH ResNet50 layout on
one ImageNet image, from the layer shapes, the tiling search of PackedConv2D::Tiling and the wire format of he_formal.tex
(input ciphertext 16 + N*109/8 bytes, output 34*used/8 + 46*N/8 bytes). Reproduces the measured 244.1 MiB (P1 -> P0,
inputs) and 234.9 MiB (P0 -> P1, outputs). Prints: per layer, the layout, bytes and occupancy against dense packing in
the same format; the per-layer choice of N = 8192; and an estimate for output repacking (dense inputs with co = 1,
outputs packed after the evaluation)."""
import math

N = 4096
IN_BYTES = 16 + N * 109 // 8


def out_bytes(u, n=N):
    return math.ceil(34 * u / 8) + 46 * n // 8


def cd(a, b):
    return -(-a // b)


# (ic, oc, H, W, k, stride, pad), H and W before padding; Cheetah_ResNet in nn/PIGEON/architectures/ResNet.hpp
L = [(3, 64, 230, 230, 7, 2, 0)]


def block(cin, mid, H, stride, down):
    out = [(cin, mid * 4, H, H, 1, stride, 0)] if down else []
    out.append((cin, mid, H, H, 1, 1, 0))
    out.append((mid, mid, H, H, 3, 1, 1) if stride == 1 else (mid, mid, H + 1, H + 1, 3, 2, 0))
    out.append((mid, mid * 4, H // stride, H // stride, 1, 1, 0))
    return out


cin = 64
for mid, n, H, s in [(64, 3, 56, 1), (128, 4, 56, 2), (256, 6, 28, 2), (512, 3, 14, 2)]:
    for i in range(n):
        L += block(cin, mid, H if i == 0 else H // s, s if i == 0 else 1, i == 0)
        cin = mid * 4
assert len(L) == 53


def reduce(ic, oc, H, W, k, s, p):  # ConvLayout::Reduced: padding, then the polyphase split of a stride
    H, W = H + 2 * p, W + 2 * p
    if s == 1:
        return ic, oc, H, W, k, k
    R = min(s, k)
    return ic * R * R, oc, cd(H, s), cd(W, s), cd(k, s), cd(k, s)


def tiling(ic, oc, H, W, kh, kw, n=N, inb=IN_BYTES, outf=out_bytes):  # PackedConv2D::Tiling, one image
    best = None
    for h in range(min(H, n), kh - 1, -1):
        for w in range(min(W, n // h), kw - 1, -1):
            for co in range(min(oc, n // h // w), 0, -1):
                ci = min(n // h // w // co, ic)
                if ci == 0:
                    continue
                sp = cd(H - kh + 1, h - kh + 1) * cd(W - kw + 1, w - kw + 1)
                used = co * (h - kh + 1) * (w - kw + 1)
                ib, ob = sp * cd(ic, ci) * inb, sp * cd(oc, co) * outf(used, n)
                if best is None or ib + ob < best[0]:
                    best = (ib + ob, ib, ob, dict(h=h, w=w, ci=ci, co=co, sp=sp))
    return best


if __name__ == "__main__":
    M = 2 ** 20
    cur, dense, n8, rp = [0, 0], [0, 0], [0, 0], [0, 0]
    stages = ["stage 1 (56x56)"] * 11 + ["stage 2 (28x28)"] * 13 + ["stage 3 (14x14)"] * 19 + ["stage 4 (7x7)"] * 10
    per_stage, keyswitch = {}, 0
    print(" #    ic    oc     HxW  k |   hxw   ci  co | in MiB out MiB | in occ out occ")
    for i, l in enumerate(L):
        ic, oc, H, W, kh, kw = reduce(*l)
        _, ib, ob, t = tiling(ic, oc, H, W, kh, kw)
        nin, nout = ic * H * W, oc * (H - kh + 1) * (W - kw + 1)
        cur[0] += ib; cur[1] += ob
        dense[0] += nin * IN_BYTES / N; dense[1] += nout * out_bytes(N) / N
        print(f"{i:2} {ic:5} {oc:5} {H:3}x{W:<3} {kh:2} | {t['h']:3}x{t['w']:<3}{t['ci']:3} {t['co']:3} | {ib/M:6.1f} {ob/M:7.1f} |"
              f" {nin / (t['sp'] * cd(ic, t['ci']) * N):6.1%} {nout / (t['sp'] * cd(oc, t['co']) * N):6.1%}")
        # N = 8192 (same 109-bit modulus) where it is cheaper
        t8 = tiling(ic, oc, H, W, kh, kw, n=8192, inb=16 + 8192 * 109 // 8)
        pick = min((t8, (0, ib, ob)), key=lambda x: x[1] + x[2]) if t8 else (0, ib, ob)
        n8[0] += pick[1]; n8[1] += pick[2]
        # repacked outputs: co = 1 layout with the fewest input bytes, outputs dense; one trace of log2(ci)
        # automorphisms per raw output ciphertext (tile x filter)
        bi = min(((sp := cd(H - kh + 1, h - kh + 1) * cd(W - kw + 1, w - kw + 1)) * cd(ic, min(N // (h * w), ic)) * IN_BYTES,
                  sp, min(N // (h * w), ic)) for h in range(min(H, N), kh - 1, -1) for w in range(min(W, N // h), kw - 1, -1))
        rp[0] += bi[0]; rp[1] += cd(nout, N) * out_bytes(N)
        keyswitch += bi[1] * oc * math.ceil(math.log2(max(bi[2], 1)))
        s = per_stage.setdefault(stages[i], [0, 0])
        s[0] += ib + ob; s[1] += bi[0] + cd(nout, N) * out_bytes(N)
    print(f"current: inputs {cur[0]/M:.1f} + outputs {cur[1]/M:.1f} = {sum(cur)/M:.1f} MiB")
    print(f"dense packing, same wire format: {dense[0]/M:.1f} + {dense[1]/M:.1f} = {sum(dense)/M:.1f} MiB")
    print(f"per-layer choice of N in (4096, 8192): {sum(n8)/M:.1f} MiB")
    print(f"output repacking: {rp[0]/M:.1f} + {rp[1]/M:.1f} = {sum(rp)/M:.1f} MiB, {keyswitch:,} key switches (traces)")
    for k, (a, b) in per_stage.items():
        print(f"  {k}: {a/M:.1f} -> {b/M:.1f} MiB")
