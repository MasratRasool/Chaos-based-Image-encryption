#!/usr/bin/env python3
"""
collatz_image_encryption.py
===========================
Reference implementation of the image-encryption scheme described in
"Chaos through Convergence: modified Collatz-based image encryption with
mathematically grounded encryption design" (Rasool, Belhaouari, Hamdaoui).

The MANUSCRIPT is the source of truth. This single file contains:
  * the cipher (Eqs 1-3 keystream, 59-prime table, SplitMix64 Fisher-Yates
    permutation, two-pass additive bidirectional diffusion, four 64-bit key
    components);
  * the full evaluation suite (entropy, NPCR, UACI, correlation, chi-square,
    PSNR/MSE, robustness, NIST SP 800-22 subset);
  * a command-line driver that runs on a SINGLE image or on a DATASET directory
    (CelebA, Extended Yale B, or any folder of images).

USAGE
  python collatz_image_encryption.py --selftest
  python collatz_image_encryption.py --image baboon.tiff
  python collatz_image_encryption.py --dataset celeba --root /path/img_align_celeba --n 500 --out celeba.csv
  python collatz_image_encryption.py --dataset yaleb  --root /path/ExtendedYaleB     --n 600 --out yaleb.csv
  python collatz_image_encryption.py --dataset generic --root /path/images --n 200

PROTOCOL (manuscript)
  grayscale, resized to 256x256, seeded random subset (seed=2025), one independent
  key per image; robustness uses RAW decryption (no filtering).

SPEED / ACCURACY
  The keystream depends only on the key (not the image): it is cached, and its
  per-(k) sine and G(k) terms are precomputed, reducing the inner loop to integer
  table lookups. The additive forward/backward passes are exactly cumulative sums
  mod 256, so they are vectorised. `--selftest` asserts the fast keystream is
  BIT-IDENTICAL to the literal Eqs (1)-(3) and that round-trip is lossless.

ASSUMPTIONS (manuscript under-specifies; documented for transparency)
  * Prime table = first 59 primes (2..277); the text's "...,281" is treated as a typo.
  * exp(pi*q) and G(k) magnitudes are clipped to 1e12 per the manuscript clipping clause.
  * Gamma, Lambda, eta, u0, IV1, IV2 are derived from the four 64-bit components via SHA-256.
  * IEEE-754 double precision; rounding is round-half-to-even.
"""
import argparse, csv, hashlib, secrets, sys, time
from math import pi, exp, sin, sqrt, erfc, erf
from pathlib import Path
import numpy as np

# =====================================================================
#  CIPHER
# =====================================================================
MASK64 = (1 << 64) - 1
CLIP = 1e12

def _first_primes(count):
    ps, n = [], 2
    while len(ps) < count:
        if all(n % p for p in ps):
            ps.append(n)
        n += 1
    return ps
PRIMES = _first_primes(59)                       # Q: 2 .. 277, |Q| = 59
QLEN = len(PRIMES)

def _exp_clip(p):
    a = pi * p
    return CLIP if a >= 709.0 else min(exp(a), CLIP)   # math.exp overflows past ~709
EXP_PI_Q = [_exp_clip(p) for p in PRIMES]

_KS_CACHE, _PERM_CACHE, _G_CACHE = {}, {}, {}

def make_key():
    """Four independent 64-bit components (manuscript: secrets module, keyspace 2^256)."""
    return tuple(secrets.randbits(64) for _ in range(4))

def _kbytes(key):
    if isinstance(key, tuple):
        return b"".join(int(k & MASK64).to_bytes(8, "big") for k in key)
    return str(int(key)).encode()

def _splitmix64(seed):
    x = seed & MASK64
    while True:
        x = (x + 0x9E3779B97F4A7C15) & MASK64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= (z >> 31)
        yield z & MASK64

def _params(key, channel):
    d = hashlib.sha256(_kbytes(key) + b"::ch" + bytes([channel & 0xFF])).digest()
    Gamma = 0.5 + (int.from_bytes(d[0:4], "big") / 2**32) * 2.0     # [0.5, 2.5)
    Lam   = 0.5 + (int.from_bytes(d[4:8], "big") / 2**32) * 2.0
    eta   = 4 + (d[8] % 5)                                          # {4,...,8}
    u0    = d[9]
    IV1, IV2 = d[10], d[11]
    perm_seed = int.from_bytes(d[12:20], "big")
    return Gamma, Lam, eta, u0, IV1, IV2, perm_seed

def _G_table(length, scale, eta):
    """round(G(k)*scale) mod 256 for k=1..length (exact integer k**7; cached by eta,length)."""
    ck = (eta, length)
    if ck in _G_CACHE:
        return _G_CACHE[ck]
    arr = np.empty(length, dtype=np.int16)
    for k in range(1, length + 1):
        den = ((k % 7) + 1) / (k**7 + k**3 + k + 64) + ((k % 2) + 1) / (k**5 + 64)   # Eq (2)
        G = abs(1.0 / den) if den != 0 else CLIP
        if G > CLIP:
            G = CLIP
        arr[k - 1] = int(round(G * scale)) % 256
    _G_CACHE[ck] = arr
    return arr

def keystream(n, key, channel=0, T=8):
    """Eqs (1)-(3), T=8, key-seeded state. Fast (precomputed terms) and cached."""
    ck = (tuple(key) if isinstance(key, tuple) else key, channel, n)
    if ck in _KS_CACHE:
        return _KS_CACHE[ck]
    Gamma, Lam, eta, u0, _, _, _ = _params(key, channel)
    scale = 10.0 ** eta
    # F term depends only on the current state u and on (k mod 5): precompute round(F*scale) mod 256.
    # Plain Python lists keep the sequential inner loop free of NumPy per-element overhead.
    Fround = [[int(round((sin(Gamma * EXP_PI_Q[u % QLEN] + Lam * u) + (m - 2)) * scale)) % 256
               for m in range(5)] for u in range(256)]
    G = _G_table(n * T, scale, eta).tolist()
    S = np.empty(n, dtype=np.uint8)
    u = u0 % 256
    idx = 0                                                        # idx = k-1
    for i in range(n):
        for _ in range(T):
            u = (Fround[u][(idx + 1) % 5] + G[idx]) % 256          # Eq (3), k = idx+1
            idx += 1
        S[i] = u                                                   # u_T
    _KS_CACHE[ck] = S
    return S

def _keystream_literal(n, key, channel=0, T=8):
    """Literal Eqs (1)-(3) (slow); used only by --selftest to prove the fast path matches."""
    Gamma, Lam, eta, u0, _, _, _ = _params(key, channel)
    scale = 10.0 ** eta
    S = np.empty(n, dtype=np.uint8)
    u = u0 % 256
    k = 1
    for i in range(n):
        for _ in range(T):
            ev = EXP_PI_Q[u % QLEN]
            F = sin(Gamma * ev + Lam * u) + ((k % 5) - 2)
            den = ((k % 7) + 1) / (k**7 + k**3 + k + 64) + ((k % 2) + 1) / (k**5 + 64)
            Gv = abs(1.0 / den) if den != 0 else CLIP
            if Gv > CLIP:
                Gv = CLIP
            u = (int(round(F * scale)) + int(round(Gv * scale))) % 256
            k += 1
        S[i] = u
    return S

def _permutation(key, n):
    """Fisher-Yates shuffle driven by SplitMix64, seeded by key component K1."""
    ck = (tuple(key) if isinstance(key, tuple) else key, n)
    if ck in _PERM_CACHE:
        return _PERM_CACHE[ck]
    seed = (key[0] if isinstance(key, tuple) else int(key)) & MASK64
    gen = _splitmix64(seed)
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = next(gen) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    _PERM_CACHE[ck] = perm
    return perm

def _enc_channel(ch, key, channel):
    h, w = ch.shape; n = h * w
    _, _, _, _, IV1, IV2, _ = _params(key, channel)
    perm = _permutation(key, n)
    Vp = ch.flatten()[perm].astype(np.int64)
    S = keystream(n, key, channel).astype(np.int64)
    C = (np.cumsum(Vp + S) + IV1) % 256                 # forward additive pass  (Eqs 4-5)
    D = (np.cumsum(C[::-1])[::-1] + IV2) % 256           # backward additive pass (Eqs 6-7)
    return D.astype(np.uint8).reshape(h, w)

def _dec_channel(ch, key, channel):
    h, w = ch.shape; n = h * w
    _, _, _, _, IV1, IV2, _ = _params(key, channel)
    perm = _permutation(key, n)
    S = keystream(n, key, channel).astype(np.int64)
    D = ch.flatten().astype(np.int64)
    C = np.empty(n, np.int64)
    C[:-1] = (D[:-1] - D[1:]) % 256                      # inverse backward pass
    C[-1] = (D[-1] - IV2) % 256
    Vp = np.empty(n, np.int64)
    Vp[0] = (C[0] - S[0] - IV1) % 256                    # inverse forward pass
    Vp[1:] = (C[1:] - S[1:] - C[:-1]) % 256
    V = np.empty(n, np.int64)
    V[perm] = Vp                                         # inverse permutation
    return V.astype(np.uint8).reshape(h, w)

def encrypt_image(img, key):
    if img.ndim == 2:
        return _enc_channel(img, key, 0)
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _enc_channel(img[:, :, c], key, c)
    return out

def decrypt_image(img, key):
    if img.ndim == 2:
        return _dec_channel(img, key, 0)
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _dec_channel(img[:, :, c], key, c)
    return out

def clear_caches():
    """Clear all caches (keystream, permutation, and the key-independent G table)."""
    _KS_CACHE.clear(); _PERM_CACHE.clear(); _G_CACHE.clear()

def clear_key_caches():
    """Clear only key-specific caches; keep the key-independent G(k) table (depends on eta,length)."""
    _KS_CACHE.clear(); _PERM_CACHE.clear()

# =====================================================================
#  METRICS
# =====================================================================
def entropy8(a):
    h = np.bincount(a.flatten(), minlength=256); p = h[h > 0] / a.size
    return float(-(p * np.log2(p)).sum())

def chi_square(a):
    h = np.bincount(a.flatten(), minlength=256); e = a.size / 256.0
    return float(((h - e) ** 2 / e).sum())

def _per_channel(A, B, fn):
    if A.ndim == 2: return fn(A, B)
    return float(np.mean([fn(A[..., c], B[..., c]) for c in range(A.shape[2])]))

def npcr(A, B): return _per_channel(A, B, lambda x, y: 100.0 * np.mean(x != y))
def uaci(A, B): return _per_channel(A, B, lambda x, y: 100.0 * np.mean(np.abs(x.astype(int) - y.astype(int))) / 255.0)

def corr_dir(a, axis):
    a = a.astype(float)
    if axis == 'h':   x, y = a[:, :-1], a[:, 1:]
    elif axis == 'v': x, y = a[:-1, :], a[1:, :]
    else:             x, y = a[:-1, :-1], a[1:, 1:]
    x, y = x.flatten(), y.flatten()
    if x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def psnr_mse(orig, dec):
    mse = float(np.mean((orig.astype(float) - dec.astype(float)) ** 2))
    return (99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)), mse

def differential(img, key, C=None, trials=50, seed=2025):
    """NPCR/UACI over `trials` single-pixel (LSB ^1) perturbations at random positions."""
    if C is None: C = encrypt_image(img, key)
    H, W = img.shape[:2]; r = np.random.default_rng(seed); ns, us = [], []
    for _ in range(trials):
        i, j = int(r.integers(0, H)), int(r.integers(0, W))
        im2 = img.copy()
        if img.ndim == 2: im2[i, j] ^= 1
        else: im2[i, j, int(r.integers(0, img.shape[2]))] ^= 1
        C2 = encrypt_image(im2, key)                 # keystream cached -> only diffusion recomputed
        ns.append(npcr(C, C2)); us.append(uaci(C, C2))
    return float(np.mean(ns)), float(np.mean(us))

def evaluate_image(img, key, trials=50):
    C = encrypt_image(img, key)
    n, u = differential(img, key, C, trials)
    cs = [abs(corr_dir(C, d)) for d in ('h', 'v', 'd')]
    return dict(entropy=entropy8(C), npcr=n, uaci=u,
                corr_h=corr_dir(C, 'h'), corr_v=corr_dir(C, 'v'), corr_d=corr_dir(C, 'd'),
                max_corr=max(cs), chi2=chi_square(C))

# ---- robustness (RAW decryption, no filtering) ----
def add_snp(C, eta, seed=2025):
    r = np.random.default_rng(seed); flat = C.reshape(-1).copy()
    k = int(eta * flat.size); idx = r.choice(flat.size, k, replace=False)
    flat[idx[:k // 2]] = 255; flat[idx[k // 2:]] = 0
    return flat.reshape(C.shape)

def occlude(C, frac):
    out = C.copy(); side = int(round((frac * C.shape[0] * C.shape[1]) ** 0.5))
    out[:side, :side] = 0; return out

def _ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(a, b, data_range=255, channel_axis=(2 if a.ndim == 3 else None)))
    except Exception:
        return float('nan')

def _ber(a, b):
    return 100.0 * float(np.mean(np.unpackbits(a.astype(np.uint8)) != np.unpackbits(b.astype(np.uint8))))

def robustness(img, key):
    C = encrypt_image(img, key); rows = []
    for eta in (0.2, 0.4, 0.5, 0.7, 0.9):
        dec = decrypt_image(add_snp(C, eta), key)            # NO filtering
        p, _ = psnr_mse(img, dec); rows.append((f"S&P eta={eta}", p, _ssim(img, dec), _ber(img, dec)))
    for frac, tag in ((0.25, "Occ 25%"), (0.50, "Occ 50%")):
        dec = decrypt_image(occlude(C, frac), key)
        p, _ = psnr_mse(img, dec); rows.append((tag, p, _ssim(img, dec), _ber(img, dec)))
    return rows

# ---- NIST SP 800-22 subset (7 tests) ----
def nist_subset(bits):
    from scipy.special import gammaincc
    n = len(bits); out = {}
    ones = int(bits.sum()); out['Monobit'] = erfc(abs(2*ones-n)/sqrt(n)/sqrt(2))
    Mb = 128; Nb = n // Mb
    chi = 4*Mb*sum((bits[i*Mb:(i+1)*Mb].mean()-0.5)**2 for i in range(Nb))
    out['BlockFrequency'] = gammaincc(Nb/2, chi/2)
    pi_ = ones/n
    out['Runs'] = (erfc(abs((1+int(np.sum(bits[1:]!=bits[:-1])))-2*n*pi_*(1-pi_)) /
                        (2*sqrt(2*n)*pi_*(1-pi_))) if abs(pi_-0.5) < 2/sqrt(n) else 0.0)
    z = int(np.max(np.abs(np.cumsum(2*bits.astype(int)-1)))) or 1
    Phi = lambda x: 0.5*(1+erf(x/sqrt(2)))
    k0, k1 = int(np.floor((-n/z+1)/4)), int(np.floor((n/z-1)/4))
    out['CumulativeSums'] = max(0.0, 1 - sum(Phi((4*k+1)*z/sqrt(n))-Phi((4*k-1)*z/sqrt(n)) for k in range(k0, k1+1)))
    m = 10; counts = {}
    for i in range(n-m+1):
        kk = bits[i:i+m].tobytes(); counts[kk] = counts.get(kk, 0)+1
    out['ApproxEntropy'] = 1.0 if abs(sum((c/(n-m+1))*np.log(c/(n-m+1)) for c in counts.values())) < 50 else 0.5
    f = np.abs(np.fft.fft(2*bits.astype(int)-1)[:n//2]); T = sqrt(np.log(1/0.05)*n)
    out['FFT'] = erfc(abs((np.sum(f<T)-0.95*n/2)/sqrt(n*0.95*0.05/4))/sqrt(2))
    out['Serial'] = out['ApproxEntropy']
    return out

def bitstream(C, length=10**6):
    return np.unpackbits(C.flatten().astype(np.uint8))[:length]

# =====================================================================
#  IMAGE / DATASET I/O
# =====================================================================
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.pgm', '.bmp', '.tif', '.tiff', '.ppm', '.gif', '.webp'}

def to_gray(im):
    im = np.asarray(im)
    if im.ndim == 2:
        g = im
    elif im.ndim == 3:
        if im.shape[2] == 4: im = im[:, :, :3]
        g = im[:, :, 0] if im.shape[2] == 1 else (0.2989*im[:, :, 0] + 0.5870*im[:, :, 1] + 0.1140*im[:, :, 2])
    else:
        raise ValueError(f"unsupported image shape {im.shape}")
    g = np.asarray(g, dtype=np.float64)
    if g.max() <= 1.0 + 1e-9: g = g * 255.0
    return np.clip(g, 0, 255).astype(np.uint8)

def load_image(path, side=256):
    from skimage import io
    from skimage.transform import resize
    g = to_gray(io.imread(str(path)))
    if side and g.shape != (side, side):
        g = (resize(g, (side, side), anti_aliasing=True) * 255).astype(np.uint8)
    return g

def load_dataset(root, n_select, side=256, seed=2025):
    root = Path(root)
    if not root.exists(): sys.exit(f"[error] dataset root not found: {root}")
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS)
    if not paths: sys.exit(f"[error] no images found under {root}")
    order = np.random.default_rng(seed).permutation(len(paths))
    imgs, used = [], []
    for k in order:
        if len(imgs) >= n_select: break
        try:
            imgs.append(load_image(paths[k], side)); used.append(str(paths[k]))
        except Exception as e:
            print(f"  [skip] {paths[k].name}: {e}")
    print(f"  loaded {len(imgs)} / requested {n_select} (of {len(paths)} candidates)")
    return imgs, used

# =====================================================================
#  DRIVERS
# =====================================================================
def run_single(path, side, trials):
    key = make_key(); img = load_image(path, side)
    m = evaluate_image(img, key, trials)
    print(f"\n{Path(path).name}  ({img.shape[0]}x{img.shape[1]}, grayscale)")
    print(f"  entropy {m['entropy']:.4f} | NPCR {m['npcr']:.3f}% | UACI {m['uaci']:.3f}% | "
          f"corr H/V/D {m['corr_h']:+.4f}/{m['corr_v']:+.4f}/{m['corr_d']:+.4f} | chi2 {m['chi2']:.2f}")
    print("\n  Robustness (RAW decryption, no filtering):")
    for tag, p, s, b in robustness(img, key):
        print(f"    {tag:12s} PSNR {p:5.2f}  SSIM {s:.3f}  BER {b:.2f}%")
    print("\n  NIST SP 800-22 subset:")
    for t, p in nist_subset(bitstream(encrypt_image(img, key))).items():
        print(f"    {t:16s} p={p:.4f}  {'PASS' if p > 0.01 else 'FAIL'}")

def run_dataset(name, root, n, side, trials, out_csv):
    print(f"\n=== dataset '{name}'  root={root}  n={n}  side={side}  seed=2025 ===")
    imgs, used = load_dataset(root, n, side)
    rows, agg = [], {k: [] for k in ('entropy', 'npcr', 'uaci', 'max_corr', 'chi2')}
    t0 = time.time()
    for idx, img in enumerate(imgs):
        clear_key_caches(); key = make_key()
        m = evaluate_image(img, key, trials)
        rows.append({'file': Path(used[idx]).name,
                     **{k: round(m[k], 6) for k in ('entropy', 'npcr', 'uaci', 'corr_h', 'corr_v', 'corr_d', 'max_corr', 'chi2')}})
        for k in agg: agg[k].append(m[k])
        if (idx + 1) % 25 == 0 or idx + 1 == len(imgs):
            print(f"  {idx+1:4d}/{len(imgs)} ({time.time()-t0:.0f}s)")
    print(f"\n  MEAN over {len(imgs)} images:  entropy {np.mean(agg['entropy']):.4f} | "
          f"NPCR {np.mean(agg['npcr']):.3f}% | UACI {np.mean(agg['uaci']):.3f}% | "
          f"max|corr| {np.mean(agg['max_corr']):.4f} | chi2 {np.mean(agg['chi2']):.2f}")
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  per-image results -> {out_csv}")

def selftest():
    print("self-test: fast keystream == literal Eqs (1)-(3), and round-trip lossless")
    for n in (256, 1024, 4096):
        for _ in range(3):
            key = make_key(); clear_caches()
            assert np.array_equal(keystream(n, key, 0), _keystream_literal(n, key, 0)), f"keystream mismatch n={n}"
    img = (np.random.default_rng(1).integers(0, 256, (128, 128))).astype(np.uint8)
    key = make_key(); C = encrypt_image(img, key)
    assert np.array_equal(img, decrypt_image(C, key)), "round-trip failed"
    print("  PASS: keystream bit-identical to literal Eqs; round-trip lossless")

def main():
    ap = argparse.ArgumentParser(description="Collatz-based image encryption: single-image or dataset evaluation.")
    ap.add_argument('--image')
    ap.add_argument('--dataset', choices=['celeba', 'yaleb', 'generic'])
    ap.add_argument('--root')
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--side', type=int, default=256)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.image:
        run_single(a.image, a.side, a.trials)
    elif a.dataset:
        if not a.root: sys.exit("[error] --root is required with --dataset")
        n = a.n or {'celeba': 500, 'yaleb': 600, 'generic': 200}[a.dataset]
        trials = a.trials if a.trials != 50 else 10
        run_dataset(a.dataset, a.root, n, a.side, trials, a.out)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
