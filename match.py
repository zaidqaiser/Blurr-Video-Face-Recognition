"""
Compare a reference picture with the average face vector from the video,
then save a collage: reference picture + a random video frame.

Usage:
    python match.py --ref reference.jpg --video sample.mp4
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_HEIGHT = 500


def make_panel(image, box, color, caption):
    """Draw a face box on the image, resize to a fixed height, add a caption strip."""
    out = image.copy()
    x1, y1, x2, y2 = box
    thick = max(2, out.shape[0] // 150)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

    scale = PANEL_HEIGHT / out.shape[0]
    out = cv2.resize(out, (int(out.shape[1] * scale), PANEL_HEIGHT))

    strip = np.full((50, out.shape[1], 3), 30, np.uint8)
    cv2.putText(strip, caption, (10, 33), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([out, strip])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Reference picture (jpg/png)")
    p.add_argument("--video", help="Video file, used to pick a random frame for the collage")
    p.add_argument("--avg", default="average_vector.npy",
                   help="Average vector saved by average_to_image.py")
    p.add_argument("--vectors", default="embeddings.npy",
                   help="Per-frame vectors saved by main.py")
    p.add_argument("--meta", default="embeddings_meta.json",
                   help="Per-frame metadata saved by main.py")
    p.add_argument("--out", default="match_result.png",
                   help="Where to save the collage picture")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Distance below this counts as a match (default 1.0)")
    p.add_argument("--seed", type=int, default=None,
                   help="Fix the random frame choice (optional)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # ---- Reference picture -> vector ----
    img = cv2.imread(args.ref)
    if img is None:
        raise SystemExit(f"Could not read image: {args.ref}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)
    if boxes is None:
        raise SystemExit("No face found in the reference picture.")

    best = int(np.argmax(probs))
    x1, y1, x2, y2 = [max(int(v), 0) for v in boxes[best]]
    face = cv2.resize(rgb[y1:y2, x1:x2], (160, 160))

    t = torch.from_numpy(face).permute(2, 0, 1).float()
    t = ((t - 127.5) / 128.0).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_vec = embedder(t).cpu().numpy()[0]
    ref_vec = ref_vec / np.linalg.norm(ref_vec)

    # ---- Compare with the average video vector ----
    avg_vec = np.load(args.avg)
    avg_vec = avg_vec / np.linalg.norm(avg_vec)

    cosine = float(np.dot(ref_vec, avg_vec))
    distance = float(np.linalg.norm(ref_vec - avg_vec))
    matched = distance < args.threshold

    print(f"Cosine similarity  : {cosine:.4f}  (higher = more similar)")
    print(f"Euclidean distance : {distance:.4f}  (lower = more similar)")
    print("Result: LIKELY THE SAME PERSON" if matched else "Result: LIKELY A DIFFERENT PERSON")
    print("Note: 1.0 is a common starting threshold; tune it on your own data.")

    green, red = (0, 200, 0), (0, 0, 255)
    result_color = green if matched else red

    # ---- Build the collage ----
    panels = [make_panel(img, (x1, y1, x2, y2), result_color, "Reference picture")]

    if args.video:
        if not (os.path.exists(args.meta) and os.path.exists(args.vectors)):
            print("Skipping video frame: run main.py first to create the vectors/meta files.")
        else:
            with open(args.meta) as f:
                meta = json.load(f)
            vectors = np.load(args.vectors)

            i = random.randrange(len(meta))          # random face from the video
            m = meta[i]
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame"])
            ok, frame = cap.read()
            cap.release()

            if ok:
                v = vectors[i] / np.linalg.norm(vectors[i])
                d = float(np.linalg.norm(ref_vec - v))
                color = green if d < args.threshold else red
                caption = f"Video frame {m['frame']} | dist {d:.2f}"
                panels.append(make_panel(frame, m["box"], color, caption))
                print(f"Random frame {m['frame']}: distance to reference {d:.4f}")
            else:
                print("Could not read the chosen video frame.")

    collage = cv2.hconcat(panels)

    banner = np.full((60, collage.shape[1], 3), result_color, np.uint8)
    text = f"{'MATCH' if matched else 'NO MATCH'} | avg distance {distance:.2f} | cosine {cosine:.2f}"
    cv2.putText(banner, text, (10, 40), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, np.vstack([banner, collage]))
    print(f"Saved collage -> {args.out}")


if __name__ == "__main__":
    main()