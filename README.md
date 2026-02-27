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


