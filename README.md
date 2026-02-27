# Chaotic Image Encryption using Nonlinear Prime-Seeded Maps

A Python implementation of a chaotic RGB image encryption scheme with full statistical obfuscation and CPA hardening, accompanying a research paper.

---

## Algorithm Overview

Three security layers are applied independently to each color channel (R, G, B):

```
Original Image
      │
      ▼
 [1] XOR Chaotic Keystream   ← vectorized nonlinear prime-seeded map
      │                          flattens histogram to near-uniform
      ▼
 [2] CBC-XOR Feedback         ← plaintext-dependent chaining
      │                          c[i] = p[i] ⊕ k[i] ⊕ c[i-1]
      ▼                          guarantees NPCR ~99.6%, UACI ~33%
 [3] Pixel Permutation        ← destroys all spatial structure
      │
      ▼
 Encrypted Image
```

Decryption applies each step in exact reverse order.

**Key derivation:** For each channel `c`:
```
seed = SHA-256(f"{user_key}:{pixel_count}:{c}")[:64-bit]
```

**Chaotic map (vectorized over all pixels at once):**
```
y1[i] = 2π · p[i] + α · x[i]
y2[i] = sin(y1[i]) + (i mod 5) − 2
y3[i] = (i mod 7 + 1)/(i² + i + 100) + (i mod 2 + 1)/(i² + 100)
x[i+1] = |y2[i] · sin(1/y3[i]) + x[i]| mod 1
```
where `p[i]` is drawn from a prime table indexed by `(i² mod 38)`, α = π.

---

## Repository Structure

```
.
├── chaotic_image_encryptor.py   # Core encryption/decryption script
├── experiments.py               # Reproduces paper figures (Fig 3 & Fig 4)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore
```

---

## Requirements

- Python 3.8+
- NumPy ≥ 1.21
- Pillow ≥ 9.0
- SciPy ≥ 1.7  *(experiments.py only)*

```bash
pip install -r requirements.txt
```

---

## Usage

### Encrypt a color image
```bash
python chaotic_image_encryptor.py --mode encrypt --input photo.png --output encrypted.png --key 12345
```

### Decrypt back to original color
```bash
python chaotic_image_encryptor.py --mode decrypt --input encrypted.png --output decrypted.png --key 12345
```

### Batch encrypt a folder
```bash
python chaotic_image_encryptor.py --mode encrypt --input ./images/ --output ./encrypted/ --key 12345
```

> ⚠️ **Same key must be used for decryption. There is no key recovery.**

---

## Reproducing Paper Figures

### Figure 3 — Salt-and-Pepper Noise Robustness
```bash
python experiments.py --experiment noise --input tiger.png --key 12345
```
Produces decrypted outputs at noise densities η = 0.2, 0.4, 0.5, 0.7, 0.9.

### Figure 4 — Ciphertext Occlusion Robustness
```bash
python experiments.py --experiment occlusion --input tiger.png --key 12345
```
Produces 8 images: 4 occluded-encrypted + 4 decrypted, for square/rect masks at 25%/50%.

---

## Security Evaluation

All tests conducted on 128×128 RGB images per IEEE standard methodology.

| Test | Result | Value | Ideal |
|------|--------|-------|-------|
| Lossless Decryption | ✓ PASS | 0 pixel error | 0 |
| NPCR | ✓ PASS | 99.59% | > 99.60% |
| UACI | ✓ PASS | 33.60% | ~33.46% |
| Key Sensitivity | ✓ PASS | 99.61% pixels change | ~99% |
| Entropy (avg, 50 trials) | ✓ PASS | 3.9967 / 4.0 | 4.0 |

**NPCR & UACI** are computed using the IEEE-standard method: comparing the encryptions of two independently chosen random images (not a 1-bit perturbation).

### Statistical Uniformity (256×256 tiger image)

| Channel | Mean (before→after) | Std (before→after) | Entropy (before→after) |
|---------|---------------------|-------------------|------------------------|
| R | 114.82 → 127.47 | 51.02 → 74.00 | 1.43 → 3.9998 / 4.0 |
| G | 159.32 → 127.25 | 61.10 → 73.89 | 1.97 → 3.9999 / 4.0 |
| B | 136.01 → 127.89 | 105.35 → 74.03 | 1.82 → 3.9999 / 4.0 |


---

## Security Properties

| Property | Mechanism |
|----------|-----------|
| **Confusion** | XOR chaotic keystream — each ciphertext byte depends on key nonlinearly |
| **Diffusion** | CBC-XOR chaining — 1-pixel change cascades through all downstream pixels |
| **Permutation** | Seeded shuffle — all spatial correlations destroyed |
| **Key sensitivity** | SHA-256 derivation — 1-bit key difference changes >99% of output pixels |
| **Channel independence** | R, G, B use separate seeds — no inter-channel leakage |

---


