#!/usr/bin/env python3
"""Train a nano neural LM for the boot sector, quantize to int8,
and simulate the EXACT integer arithmetic the asm will perform.

Everything the boot sector does is defined here first; boot.asm must match.
"""
import json
import numpy as np

rng = np.random.default_rng(7)

# ---------------- architecture (frozen) ----------------
V = 64      # vocab (6-bit indices)
C = 8       # context chars
E = 6       # embedding dim
H = 96      # hidden units
Q = 12      # activation fixed-point (Q12)

STEPS = 40000
BATCH = 512
LR = 0.008

# ---------------- data ----------------
text = open("corpus.txt").read()
freq = {}
for ch in text:
    freq[ch] = freq.get(ch, 0) + 1
top = sorted(freq.items(), key=lambda kv: -kv[1])
vocab = [ch for ch, _ in top[: V - 1]]
if "\n" not in vocab:
    vocab[-1] = "\n"
vocab.append("\n")
assert len(vocab) == V, len(vocab)
vmap = {ch: i for i, ch in enumerate(vocab)}
space_id = vmap[" "]

ids = np.array([vmap.get(ch, space_id) for ch in text], dtype=np.int64)
print("corpus chars:", len(ids), "vocab:", "".join(
    c if c != "\n" else "\\n" for c in vocab))

# ---------------- model ----------------
emb = rng.normal(0, 0.1, (V, E))
W1 = rng.normal(0, 0.2, (H, C * E))   # [j][i]
b1 = np.zeros(H)
W2 = rng.normal(0, 0.2, (V, H))       # [v][h]
b2 = np.zeros(V)

params = [emb, W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def softclip(a):          # activation used by BOTH training and asm
    return a / (1.0 + np.abs(a))


def forward(x_idx):
    xin = emb[x_idx].reshape(len(x_idx), C * E)      # [B, 30]
    a1 = xin @ W1.T + b1
    h = softclip(a1)
    logits = h @ W2.T + b2
    return xin, a1, h, logits


losses = []
for step in range(1, STEPS + 1):
    pos = rng.integers(0, len(ids) - C - 1, BATCH)
    x_idx = np.stack([ids[pos + k] for k in range(C)], axis=1)   # [B,C]
    y = ids[pos + C]

    xin, a1, h, logits = forward(x_idx)
    p = softmax(logits)
    loss = -np.log(p[np.arange(BATCH), y] + 1e-9).mean()
    losses.append(loss)

    dlogits = p
    dlogits[np.arange(BATCH), y] -= 1
    dlogits /= BATCH

    gW2 = dlogits.T @ h
    gb2 = dlogits.sum(0)
    dh = dlogits @ W2
    da1 = dh * (1.0 / (1.0 + np.abs(a1)) ** 2)     # softclip grad
    gW1 = da1.T @ xin
    gb1 = da1.sum(0)
    gemb_flat = (da1 @ W1)                          # [B, 30]
    gemb = np.zeros_like(emb)
    np.add.at(gemb, x_idx.reshape(-1), gemb_flat.reshape(-1, E))

    lr = LR * (0.1 ** (step / STEPS))
    for i, g in enumerate([gemb, gW1, gb1, gW2, gb2]):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g * g
        mh = m[i] / (1 - beta1 ** step)
        vh = v[i] / (1 - beta2 ** step)
        params[i] -= lr * mh / (np.sqrt(vh) + eps)

    if step % 2500 == 0:
        print(f"step {step:6d}  loss {np.mean(losses[-2500:]):.4f}")

emb, W1, b1, W2, b2 = params
np.savez("model.npz", emb=emb, W1=W1, b1=b1, W2=W2, b2=b2,
         vocab=np.array("".join(vocab)))

# ---------------- float baseline sampler ----------------
def gen_float(n, temp=0.9, seed_ctx=None):
    ctx = list(seed_ctx if seed_ctx is not None else
               [vmap.get(c, space_id) for c in "[   0.00"])
    out = []
    rs = np.random.default_rng(0)
    for _ in range(n):
        _, _, _, logits = forward(np.array([ctx]))
        pr = softmax(logits[0] / temp)
        nxt = rs.choice(V, p=pr)
        out.append(vocab[nxt])
        ctx = ctx[1:] + [nxt]
    return "".join(out)


print("\n--- FLOAT baseline sample ---")
print(gen_float(500))

# ---------------- quantization ----------------
LQ = 8    # logit fixed-point Q8 (logits ±25 * 256 fits int16)


def pow2_scale(w):
    """scale = 2^e, largest e such that max|w*2^e| <= 127"""
    mx = np.abs(w).max()
    e = int(np.floor(np.log2(127.0 / mx))) if mx > 0 else 7
    while np.abs(w * (1 << e)).max() > 127:
        e -= 1
    return e


def q_int8(w, e):
    return np.clip(np.round(w * (1 << e)), -127, 127).astype(np.int8)


# rescale emb/W1 (function-preserving) to equalize relative quantization
# precision of both tensors, then use max-usage power-of-two scales.
alpha = np.percentile(np.abs(emb), 95) / np.percentile(np.abs(W1), 95)
emb = emb * alpha
W1 = W1 / alpha

ee = pow2_scale(emb)          # emb scale 2^ee
ew1 = pow2_scale(W1)
ew2 = pow2_scale(W2)

emb_q = q_int8(emb, ee)
W1_q = q_int8(W1, ew1)        # stored [j][i]
W2_q = q_int8(W2, ew2)        # stored [v][h]


def q_fix(w, q):
    return np.clip(np.round(w * (1 << q)), -32700, 32700).astype(np.int16)


b1_q = q_fix(b1, Q)           # preact Q12
b2_q = q_fix(b2, LQ)          # logits Q8

# shifts may be negative: asm uses sar for >=0, shl for <0 (32-bit math,
# preacts/logits are bounded by the model, not by int16)
A1SHIFT = ee + ew1 - Q
A2SHIFT = 6 + ew2 - LQ
nclip1 = int((np.abs(W1_q) == 127).sum() + (np.abs(emb_q) == 127).sum())
nclip2 = int((np.abs(W2_q) == 127).sum())
print(f"\nscales: emb 2^{ee}  w1 2^{ew1}  w2 2^{ew2}  "
      f"a1shift {A1SHIFT}  a2shift {A2SHIFT}  "
      f"sat(emb,W1) {nclip1} sat(W2) {nclip2}")

# ---------------- EXACT integer simulation of boot.asm ----------------
import math

emb_f = emb_q.reshape(-1)          # [v*E + r]
W1_f = W1_q.reshape(-1).astype(np.int64)   # [j*(C*E) + i]
W2_f = W2_q.reshape(-1).astype(np.int64)   # [v*H + h]
W1_m = W1_f.reshape(H, C * E)
W2_m = W2_f.reshape(V, H)
b1_i = b1_q.astype(np.int64)
b2_i = b2_q.astype(np.int64)


def trunc_div(a, b):          # x86 idiv semantics
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


tvec = np.vectorize(trunc_div)


class BootSim:
    """Mirrors boot.asm arithmetic instruction-for-instruction.
    Integer dot products are exact regardless of summation order,
    so vectorized numpy here == loops in asm."""

    def __init__(self, noise_scale, seed=0x1234):
        self.NS = noise_scale
        self.rnd = seed & 0xFFFF or 1

    def lcg(self):
        self.rnd = (self.rnd * 25173 + 13849) & 0xFFFF
        return self.rnd

    def logits_for(self, ctx):
        xin = np.array([emb_f[ctx[k] * E + r]
                        for k in range(C) for r in range(E)],
                       dtype=np.int64)
        a1 = W1_m @ xin                                    # exact int64
        p = (a1 >> A1SHIFT) if A1SHIFT >= 0 else (a1 << -A1SHIFT)
        p = p + b1_i                                       # Q12, int32 range
        y = tvec(p * 64, 4096 + np.abs(p))
        y = np.clip(y, -64, 64)
        a2 = W2_m @ y                                      # exact int64
        return ((a2 >> A2SHIFT) if A2SHIFT >= 0
                else (a2 << -A2SHIFT)) + b2_i              # Q8

    def gen(self, n, ctx0):
        ctx = list(ctx0)
        out = []
        for _ in range(n):
            logits = self.logits_for(ctx)
            best, bi = -(1 << 30), 0
            for vi in range(V):
                r8 = (self.lcg() >> 8) & 0xFF
                if r8 >= 128:
                    r8 -= 256
                cand = int(logits[vi]) + r8 * self.NS
                if cand > best:
                    best, bi = cand, vi
            out.append(vocab[bi])
            ctx = ctx[1:] + [bi]
        return "".join(out)


seed_ctx = [vmap.get(c, space_id) for c in "[   0.00"]
NOISE_CHOICE = 8

# overflow guard: logits live in int16 in asm (Q8)
_probe = BootSim(0)
_lmax = 0
_ctx = list(seed_ctx)
for _ in range(300):
    _l = _probe.logits_for(_ctx)
    _lmax = max(_lmax, int(np.abs(_l).max()))
    _ctx = _ctx[1:] + [int(_l.argmax())]
print(f"max |logit| on greedy walk: {_lmax} (int16 limit 32767)")
assert _lmax < 30000, _lmax

print("\n--- INTEGER (asm-identical) samples, noise sweep ---")
for ns in [8, 24, 48, 96]:
    s = BootSim(ns)
    txt = s.gen(400, seed_ctx)
    print(f"\n[NS={ns}]")
    print(txt[:1200])

# ---------------- emit artifacts ----------------
blob = bytearray()
blob += bytes(ord(c) if c != "\n" else 10 for c in vocab)   # 64
blob += emb_q.tobytes()                                     # 384
blob += W1_q.tobytes()                                      # H*C*E
blob += W2_q.tobytes()                                      # V*H
blob += b1_q.tobytes()                                      # H*2
blob += b2_q.tobytes()                                      # V*2
expected = 64 + V * E + H * C * E + V * H + 2 * H + 2 * V
assert len(blob) == expected, (len(blob), expected)

# strings live on disk too, the boot sector is code only
HDR_ADDR = 0x8000 + len(blob)
blob += (b"bootgpt: a neural network dreaming a linux boot log"
         b"\r\nno os involved - press any key to reboot\r\n\x00")
BYE_ADDR = 0x8000 + len(blob)
blob += b"\r\ndream interrupted - rebooting reality\x00"

SECTORS = (len(blob) + 511) // 512
blob += b"\x00" * (SECTORS * 512 - len(blob))
open("weights.bin", "wb").write(blob)

with open("params.inc", "w") as f:
    w1off = 64 + V * E
    w2off = w1off + H * C * E
    b1off = w2off + V * H
    b2off = b1off + 2 * H
    f.write(f"; auto-generated by train.py\n"
            f"VOCAB   equ {V}\n"
            f"CTX     equ {C}\n"
            f"EMBDIM  equ {E}\n"
            f"HID     equ {H}\n"
            f"W1_OFF  equ {w1off}\n"
            f"W2_OFF  equ {w2off}\n"
            f"B1_OFF  equ {b1off}\n"
            f"B2_OFF  equ {b2off}\n"
            f"A1SHIFT equ {A1SHIFT}\n"
            f"A2SHIFT equ {A2SHIFT}\n"
            f"NOISE   equ {NOISE_CHOICE}\n"
            f"SECTORS equ {SECTORS}\n"
            f"HDRADDR equ {HDR_ADDR:#x}\n"
            f"BYEADDR equ {BYE_ADDR:#x}\n")
    f.write("%define SEEDBYTES db " + ",".join(str(i) for i in seed_ctx) + "\n")
json.dump({"A1SHIFT": A1SHIFT, "A2SHIFT": A2SHIFT,
           "scales": [ee, ew1, ew2]}, open("params.json", "w"))
print(f"\nweights.bin: {len(blob)} bytes ({SECTORS} sectors), params.inc written")
