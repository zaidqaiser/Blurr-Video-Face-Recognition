import argparse

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", default="embeddings.npy",
                   help="Vectors file created by main.py")
    p.add_argument("--out", default="average_vector",
                   help="Output file prefix")
    args = p.parse_args()

    vectors = np.load(args.vectors)            # shape: (N, 512)
    print(f"Loaded {vectors.shape[0]} vectors of size {vectors.shape[1]}")

    # 1) Average all vectors into one
    avg = vectors.mean(axis=0)                 # shape: (512,)
    avg = avg / np.linalg.norm(avg)            # re-normalise to unit length
    np.save(f"{args.out}.npy", avg)

    # 2) Turn the 512 numbers into a 16 x 32 grid of pixels
    side_h, side_w = 16, 32
    grid = avg.reshape(side_h, side_w)

    # Scale values to 0-255
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid = (grid * 255).astype(np.uint8)

    # Enlarge so it is visible and apply a colour map
    big = cv2.resize(grid, (side_w * 20, side_h * 20),
                     interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(big, cv2.COLORMAP_JET)

    cv2.imwrite(f"{args.out}.png", colored)
    print(f"Saved average vector -> {args.out}.npy")
    print(f"Saved image          -> {args.out}.png")


if __name__ == "__main__":
    main()