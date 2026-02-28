# Collatz-Based Image Encryption

Python implementation of the image encryption scheme proposed in:

> Masrat Rasool, Samir Brahim Belhaouari, Bechir Hamdaoui.  
> *Chaos through convergence: modified Collatz-based image encryption  
> with mathematically grounded encryption design.*  


## Requirements
```
pip install pillow numpy scipy
```

## Usage
```bash
# Encrypt
python collatz_image_encryption.py --mode encrypt \
    --input photo.png --output enc.png \
    --k1 11111 --k2 22222 --k3 33333 --k4 44444

# Decrypt
python collatz_image_encryption.py --mode decrypt \
    --input enc.png --output dec.png \
    --k1 11111 --k2 22222 --k3 33333 --k4 44444

# Reproduce all paper security results
python security_tests.py --input photo.png
```

## Files

| File | Description |
|---|---|
| `collatz_image_encryption.py` | Encryption/decryption library |
| `security_tests.py` | Reproduces all security metrics from the paper |

## Tested on

Ubuntu 18.04 LTS · Python 3.10 · NumPy 1.24
