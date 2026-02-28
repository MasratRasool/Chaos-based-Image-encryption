'''
Collatz-Based Image Encryption / Decryption
'''

import argparse
import sys
import secrets
import time
from math import pi, exp, sin
from pathlib import Path

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Prime table Q  (fixed, public)
# Paper: Q = {2, 3, 5, 7, 11, 13, ...}  starts from 2
# ─────────────────────────────────────────────────────────────
Q = [
      2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
     31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
     73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
]
Q_SIZE = len(Q)

T        = 8                           # iterations per pixel (paper: T = 8)
_U64_MAX = float(0xFFFFFFFFFFFFFFFF)


# ─────────────────────────────────────────────────────────────
# Key
# ─────────────────────────────────────────────────────────────
class Key:
    """
    Four independent 64-bit integers as described in the paper.

    K1 : permutation seed + IV1 for forward CBC
    K2 : Gamma   -> float in (0.5, 3.0)
    K3 : Lambda  -> float in (0.1, 2.0)
    K4 : eta     -> int   in {2, 3, 4, 5, 6}
    IV2 derived from K1 for backward CBC
    """

    def __init__(self, k1: int, k2: int, k3: int, k4: int):
        self.k1     = int(k1) & 0xFFFFFFFFFFFFFFFF
        self.k2     = int(k2) & 0xFFFFFFFFFFFFFFFF
        self.k3     = int(k3) & 0xFFFFFFFFFFFFFFFF
        self.k4     = int(k4) & 0xFFFFFFFFFFFFFFFF
        self.Gamma  = 0.5 + (self.k2 / _U64_MAX) * 2.5
        self.Lambda = 0.1 + (self.k3 / _U64_MAX) * 1.9
        self.eta    = 2   + int((self.k4 / _U64_MAX) * 4)
        self.iv1    =  self.k1        & 0xFF
        self.iv2    = (self.k1 >> 8)  & 0xFF

    @staticmethod
    def generate() -> 'Key':
        """Cryptographically random key via Python secrets module."""
        return Key(secrets.randbits(64), secrets.randbits(64),
                   secrets.randbits(64), secrets.randbits(64))

    def __repr__(self):
        return (f"Key(\n"
                f"  k1={self.k1}\n  k2={self.k2}\n"
                f"  k3={self.k3}\n  k4={self.k4}\n"
                f"  -> Gamma={self.Gamma:.6f}  Lambda={self.Lambda:.6f}"
                f"  eta={self.eta}  iv1={self.iv1}  iv2={self.iv2})")


# ─────────────────────────────────────────────────────────────
# Precompute G(k) for k=1..T  — Paper Equation (2)
# G(k) = |((k%7+1)/(k^7+k^3+k+64) + (k%2+1)/(k^5+64))^{-1}|
# Clipped to 1e12 as stated in paper.
# ─────────────────────────────────────────────────────────────
def _make_G() -> list:
    G = []
    for k in range(1, T + 1):
        kf    = float(k)
        inner = (((k % 7) + 1) / (kf**7 + kf**3 + kf + 64.0)
               + ((k % 2) + 1) / (kf**5 + 64.0))
        G.append(min(abs(1.0 / inner) if abs(inner) > 1e-300 else 1e12, 1e12))
    return G

_G = _make_G()


# ─────────────────────────────────────────────────────────────
# Chaotic stream — vectorised
#
# The stream position-seed function has period 256:
#   stream(i) = stream(i mod 256)
# because the initial state x = float(i % 256) repeats every 256 positions.
#
# Therefore:
#   1. Build a 256-entry LUT using the paper's exact equations (T=8 loops,
#      IEEE-754 double, round-to-nearest-even) — O(256 * T) scalar ops.
#   2. Apply to all N positions via NumPy index: stream = lut[arange(N) % 256]
#      — one vectorised operation, no Python loop over pixels.
#
# The LUT computation is mathematically identical to the paper's equations.
# ─────────────────────────────────────────────────────────────
def _build_stream_lut(key: Key) -> np.ndarray:
    """
    Build 256-entry lookup table of chaotic stream values.
    lut[i] = stream value for any pixel position j where j % 256 == i.
    Uses paper Equations (1)-(3) exactly, with IEEE-754 double precision
    and round-to-nearest-even throughout.
    """
    Gamma = key.Gamma
    Lambda = key.Lambda
    scale  = 10 ** key.eta
    lut    = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        x     = float(i)      # seed from position index
        total = 0

        for k in range(1, T + 1):
            # Equation (1): F(u_k)
            q_val   = float(Q[int(x) % Q_SIZE])
            exp_arg = min(pi * q_val, 700.0)           # prevent float64 overflow
            F       = sin(Gamma * exp(exp_arg) + Lambda * x) + (k % 5 - 2)
            F       = max(-1e12, min(1e12, F))          # paper: clip to 1e12

            # Equation (3): round-to-nearest-even via Python built-in round()
            delta   = (round(F * scale) + round(_G[k - 1] * scale)) % 256
            total   = (total + delta) % 256
            x       = float(delta)                     # state evolves for next k

        lut[i] = total

    return lut


def _build_stream(n: int, key: Key) -> np.ndarray:
    """
    Build full-length stream array of length n.
    Exploits period-256 property: one NumPy index op after LUT construction.
    """
    lut     = _build_stream_lut(key)
    indices = np.arange(n, dtype=np.int32) % 256
    return lut[indices]                                # vectorised, no Python loop


# ─────────────────────────────────────────────────────────────
# Bidirectional additive CBC — fully vectorised via cumsum
#
# Forward encrypt:
#   c[i] = (p[i] + s[i] + c[i-1]) mod 256,   c[-1] = IV1
#   Expanding the recurrence:
#   c[i] = (p[0]+s[0]) + (p[1]+s[1]) + ... + (p[i]+s[i]) + IV1  (mod 256)
#        = cumsum(p + s)[i] + IV1   mod 256
#   -> single np.cumsum call
#
# Backward encrypt (applied to forward output c):
#   d[i] = (c[i] + d[i+1]) mod 256,   d[N] = IV2
#   Expanding:  d[i] = c[i] + c[i+1] + ... + c[N-1] + IV2  mod 256
#             = reverse_cumsum(c)[i] + IV2   mod 256
#   -> reverse array, cumsum, reverse back
#
# Decrypt inverses:
#   Backward inv: c[i] = d[i] - d[i+1]  (shift-subtract, vectorised)
#   Forward  inv: p[i] = c[i] - s[i] - c[i-1]  (shift-subtract, vectorised)
# ─────────────────────────────────────────────────────────────
def _forward_enc(arr: np.ndarray, stream: np.ndarray, iv: int) -> np.ndarray:
    combined = arr.astype(np.int32) + stream.astype(np.int32)
    return (np.cumsum(combined) + int(iv)).astype(np.uint8)


def _forward_dec(c: np.ndarray, stream: np.ndarray, iv: int) -> np.ndarray:
    shifted      = np.empty_like(c, dtype=np.int32)
    shifted[0]   = int(iv)
    shifted[1:]  = c[:-1].astype(np.int32)
    return (c.astype(np.int32) - stream.astype(np.int32) - shifted).astype(np.uint8)


def _backward_enc(c: np.ndarray, iv: int) -> np.ndarray:
    rev_cumsum = np.cumsum(c[::-1].astype(np.int32))[::-1]
    return (rev_cumsum + int(iv)).astype(np.uint8)


def _backward_dec(d: np.ndarray, iv: int) -> np.ndarray:
    shifted      = np.empty_like(d, dtype=np.int32)
    shifted[:-1] = d[1:].astype(np.int32)
    shifted[-1]  = int(iv)
    return (d.astype(np.int32) - shifted).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Permutation layer  (Fisher-Yates via numpy default_rng, K1)
# ─────────────────────────────────────────────────────────────
def _make_perm(k1: int, n: int) -> np.ndarray:
    return np.random.default_rng(k1).permutation(n)

def _make_inv_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# ─────────────────────────────────────────────────────────────
# Single channel encrypt / decrypt
# ─────────────────────────────────────────────────────────────
def _enc_channel(ch: np.ndarray, key: Key) -> np.ndarray:
    flat   = ch.flatten().astype(np.uint8)
    n      = len(flat)
    perm   = _make_perm(key.k1, n)
    flat   = flat[perm]                                # Step 1: permutation
    stream = _build_stream(n, key)                     # Step 2: chaotic stream
    flat   = _forward_enc(flat, stream, key.iv1)       # Step 3: forward CBC
    flat   = _backward_enc(flat, key.iv2)              # Step 4: backward CBC
    return flat.reshape(ch.shape)


def _dec_channel(ch: np.ndarray, key: Key) -> np.ndarray:
    flat   = ch.flatten().astype(np.uint8)
    n      = len(flat)
    flat   = _backward_dec(flat, key.iv2)              # Reverse Step 4
    stream = _build_stream(n, key)
    flat   = _forward_dec(flat, stream, key.iv1)       # Reverse Step 3
    perm   = _make_perm(key.k1, n)
    flat   = flat[_make_inv_perm(perm)]                # Reverse Step 1
    return flat.reshape(ch.shape)


# ─────────────────────────────────────────────────────────────
# Public API — grayscale (H×W) or RGB (H×W×3)
# ─────────────────────────────────────────────────────────────
def encrypt_image(img: np.ndarray, key: Key) -> np.ndarray:
    """Encrypt grayscale (H×W) or RGB (H×W×3) uint8 image."""
    if img.ndim == 2:
        return _enc_channel(img, key)
    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        result[:, :, c] = _enc_channel(img[:, :, c], key)
    return result


def decrypt_image(img: np.ndarray, key: Key) -> np.ndarray:
    """Decrypt grayscale (H×W) or RGB (H×W×3) uint8 image."""
    if img.ndim == 2:
        return _dec_channel(img, key)
    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        result[:, :, c] = _dec_channel(img[:, :, c], key)
    return result


# ─────────────────────────────────────────────────────────────
# Statistical metrics
# ─────────────────────────────────────────────────────────────
def entropy_8bit(arr: np.ndarray) -> float:
    """Shannon entropy, 256 bins. Maximum = 8.0 bits."""
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
    total   = arr.size
    return sum(-h / total * np.log2(h / total) for h in hist if h > 0)


def npcr(c1: np.ndarray, c2: np.ndarray) -> float:
    """Number of Pixel Change Rate (%). Target ~99.6%."""
    return 100.0 * (c1 != c2).sum() / c1.size


def uaci(c1: np.ndarray, c2: np.ndarray) -> float:
    """Unified Average Changing Intensity (%). Target ~33.4%."""
    return 100.0 * np.abs(c1.astype(np.int32) - c2.astype(np.int32)).sum() \
           / (255.0 * c1.size)


def adj_corr(arr: np.ndarray, direction: str) -> float:
    """Adjacent pixel correlation. Target ~0 after encryption."""
    a = arr.astype(np.float64)
    if direction == 'H':   x, y = a[:, :-1].flatten(), a[:, 1:].flatten()
    elif direction == 'V': x, y = a[:-1, :].flatten(), a[1:, :].flatten()
    else:                  x, y = a[:-1, :-1].flatten(), a[1:, 1:].flatten()
    return float(np.corrcoef(x, y)[0, 1])


def run_security_tests(img: np.ndarray, key: Key, n_trials: int = 50) -> dict:
    """Full security evaluation matching paper protocol."""
    r   = {}
    t0  = time.time()
    enc = encrypt_image(img, key)
    r['time_ms'] = (time.time() - t0) * 1000
    r['mp_s']    = (img.shape[0] * img.shape[1] / 1e6) / (r['time_ms'] / 1000)

    chs          = [enc] if enc.ndim == 2 else [enc[:, :, c] for c in range(3)]
    r['entropy'] = float(np.mean([entropy_8bit(c) for c in chs]))

    rng          = np.random.default_rng(42)
    npcrs, uacis = [], []
    for _ in range(n_trials):
        img2 = img.copy()
        ri, rj = rng.integers(0, img.shape[0]), rng.integers(0, img.shape[1])
        if img2.ndim == 2:
            img2[ri, rj] = (int(img2[ri, rj]) + 1) % 256
        else:
            img2[ri, rj, 0] = (int(img2[ri, rj, 0]) + 1) % 256
        c2 = encrypt_image(img2, key)
        npcrs.append(npcr(enc, c2))
        uacis.append(uaci(enc, c2))

    r['npcr']   = float(np.mean(npcrs))
    r['uaci']   = float(np.mean(uacis))
    ch          = chs[0]
    r['corr_H'] = adj_corr(ch, 'H')
    r['corr_V'] = adj_corr(ch, 'V')
    r['corr_D'] = adj_corr(ch, 'D')
    r['lossless'] = bool(np.array_equal(img, decrypt_image(enc, key)))
    return r


def print_report(r: dict, name: str = "") -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Security Report{(' — ' + name) if name else ''}")
    print(sep)
    print(f"  Lossless : {'PASS  ✓' if r['lossless'] else 'FAIL  ✗'}")
    print(f"  Entropy  : {r['entropy']:.4f} / 8.0    (target > 7.990)")
    print(f"  NPCR     : {r['npcr']:.4f} %   (target > 99.5%)")
    print(f"  UACI     : {r['uaci']:.4f} %   (target ~33.4%)")
    print(f"  Corr H   : {r['corr_H']:+.6f}          (target ~0)")
    print(f"  Corr V   : {r['corr_V']:+.6f}          (target ~0)")
    print(f"  Corr D   : {r['corr_D']:+.6f}          (target ~0)")
    print(f"  Time     : {r['time_ms']:.1f} ms")
    print(f"  Throughput: {r['mp_s']:.3f} MP/s")
    print(sep)


# ─────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img if img.mode == 'L' else img.convert("RGB"), dtype=np.uint8)

def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr.astype(np.uint8),
                    'L' if arr.ndim == 2 else 'RGB').save(path)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collatz-Based Image Encryption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mode",   required=True, choices=["encrypt", "decrypt", "test"])
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--k1", type=int, default=0, help="Permutation seed (64-bit int)")
    ap.add_argument("--k2", type=int, default=0, help="Gamma key        (64-bit int)")
    ap.add_argument("--k3", type=int, default=0, help="Lambda key       (64-bit int)")
    ap.add_argument("--k4", type=int, default=0, help="Eta key          (64-bit int)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"[ERROR] Not found: {src}"); sys.exit(1)
    if src.suffix.lower() not in SUPPORTED_EXTS:
        print(f"[ERROR] Unsupported: {src.suffix}"); sys.exit(1)

    img = load_image(str(src))
    print(f"\n  Image : {src.name}  "
          f"{'grayscale' if img.ndim == 2 else 'RGB'}  shape={img.shape}")

    if args.mode == "test":
        key = Key.generate()
        print(f"\n  Key   :\n  {key}")
        print(f"\n  Running security tests (50 NPCR trials)...")
        r = run_security_tests(img, key, n_trials=50)
        print_report(r, src.name)
        status = "[PASS]" if r['lossless'] else "[FAIL]"
        print(f"\n  {status}")
        sys.exit(0 if r['lossless'] else 1)

    if args.output is None:
        print("[ERROR] --output required"); sys.exit(1)

    key = Key(args.k1, args.k2, args.k3, args.k4)
    print(f"  Key   : Gamma={key.Gamma:.4f}  Lambda={key.Lambda:.4f}  "
          f"eta={key.eta}  iv1={key.iv1}  iv2={key.iv2}")

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0  = time.time()
    out = encrypt_image(img, key) if args.mode == "encrypt" else decrypt_image(img, key)
    ms  = (time.time() - t0) * 1000

    save_image(out, str(dst))

    if args.mode == "encrypt":
        chs = [out] if out.ndim == 2 else [out[:, :, c] for c in range(3)]
        print(f"  Entropy: {np.mean([entropy_8bit(c) for c in chs]):.4f} / 8.0")

    n_px = img.shape[0] * img.shape[1]
    print(f"  Time  : {ms:.1f} ms  ({n_px/1e6/(ms/1000):.3f} MP/s)  ->  {dst}")
    print("\n[OK]")


if __name__ == "__main__":
    main()
