#!/usr/bin/env python3
import os, glob, numpy as np
from astropy.io import fits

def fits_to_npy(src_dir=".", dst_dir="npy"):
    os.makedirs(dst_dir, exist_ok=True)
    fits_files = sorted(glob.glob(os.path.join(src_dir, "*.fits")))
    if not fits_files:
        print("No FITS files found.")
        return
    for fp in fits_files:
        try:
            with fits.open(fp) as hdul:
                data = next((h.data for h in hdul if h.data is not None), None)
                if data is None:
                    print(f"Skipping {fp}: no image data")
                    continue
                data = np.squeeze(np.asarray(data))
                if data.ndim != 2:
                    print(f"Skipping {fp}: non-2D shape {data.shape}")
                    continue
                data = np.nan_to_num(data, copy=False).astype(np.float32)
                base = os.path.splitext(os.path.basename(fp))[0]
                out = os.path.join(dst_dir, base + ".npy")
                np.save(out, data)
                print(f"Saved {out}")
        except Exception as e:
            print(f"Error {fp}: {e}")

if __name__ == "__main__":
    fits_to_npy()