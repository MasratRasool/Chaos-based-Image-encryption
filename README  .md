# Collatz-based Image Encryption — Reference Implementation

Single-file reference implementation accompanying the manuscript
*"Chaos through Convergence: modified Collatz-based image encryption with
mathematically grounded encryption design."* The manuscript is the source of truth;
`collatz_image_encryption.py` implements its algorithm, protocols, and evaluation.

## Contents
- `collatz_image_encryption.py` — cipher + full evaluation suite + CLI
- `requirements.txt` — pinned dependencies
- `LICENSE` — MIT

## Environment
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Reference platform: Ubuntu 18.04 LTS, Python 3.10, NumPy 1.26.x.

## Verify the implementation
```
python collatz_image_encryption.py --selftest
```
Asserts the (fast) keystream is BIT-IDENTICAL to the literal Equations (1)-(3) and
that encryption round-trips losslessly.

## Single image
```
python collatz_image_encryption.py --image baboon.tiff
```
Reports entropy, NPCR, UACI, H/V/D correlation, chi-square, raw (no-filtering)
robustness, and the NIST SP 800-22 subset.

## Datasets (grayscale, 256x256, seeded subset = 2025, one key per image)
```
python collatz_image_encryption.py --dataset celeba --root /path/img_align_celeba --n 500 --out celeba.csv
python collatz_image_encryption.py --dataset yaleb  --root /path/ExtendedYaleB     --n 600 --out yaleb.csv
python collatz_image_encryption.py --dataset generic --root /path/images --n 200
```
The loader is recursive; it handles RGB/RGBA/grayscale/palette inputs and common
formats (jpg, png, pgm, bmp, tif, ...), skips unreadable files, and writes a
per-image CSV. A full 1,100-image run takes a few minutes.

## Reproducibility note
The keystream evaluates `sin`/`exp` (Equations 1-3) under fixed IEEE-754 double
precision with round-to-nearest-even, then reduces modulo 256. Lossless decryption
requires the encrypting and decrypting environments to produce the **same** keystream.
Bit-identical encryption/decryption is verified on the reference platform above; on a
different platform, library/`libm` differences in transcendental evaluation could in
principle alter the keystream. For cross-platform use, encrypt and decrypt with the
same NumPy/Python build, or pin the environment via `requirements.txt`.

## Data
Datasets are not redistributed. USC-SIPI benchmark images: https://sipi.usc.edu/database/ .
CelebA and Extended Yale B from their official sources, for academic use.

## NPCR/UACI protocol
NPCR and UACI use the standard single-pixel differential test (a least-significant-bit
change at a randomly selected pixel), averaged over positions, matching the values
reported in the paper.
