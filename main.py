import argparse
import json

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


def parse_args():
    p = argparse.ArgumentParser(description="Video -> face embeddings")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--out", default="embeddings", help="Output file prefix")
    p.add_argument("--skip", type=int, default=1,
                   help="Process every Nth frame (default 1 = every frame)")
    p.add_argument("--all-faces", action="store_true",
                   help="Keep every face in a frame (default: only the best one)")
    p.add_argument("--min-prob", type=float, default=0.95,
                   help="Minimum MTCNN confidence to keep a face")
    p.add_argument("--show", action="store_true",
                   help="Show a preview window (press q to quit)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # MTCNN finds faces; InceptionResnetV1 turns a face into a 512-d vector
    print("Loading models (first run downloads ~100 MB, please wait)...")
    mtcnn = MTCNN(keep_all=True, device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("Models loaded.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames. Starting...")

    vectors = []   # list of 512-d numpy arrays
    meta = []      # info about each vector (frame number, box, confidence)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect on a smaller copy (much faster on CPU), crop from original
            scale = 640 / rgb.shape[1] if rgb.shape[1] > 640 else 1.0
            small = cv2.resize(rgb, None, fx=scale, fy=scale) if scale < 1 else rgb
            boxes, probs = mtcnn.detect(small)
            if boxes is not None and scale < 1:
                boxes = boxes / scale

            if boxes is not None and not args.all_faces:
                best = int(np.argmax(probs))
                boxes, probs = boxes[best:best + 1], probs[best:best + 1]

            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob is None or prob < args.min_prob:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    face = rgb[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Prepare face for the embedding model
                    face = cv2.resize(face, (160, 160))
                    t = torch.from_numpy(face).permute(2, 0, 1).float()
                    t = (t - 127.5) / 128.0
                    t = t.unsqueeze(0).to(device)

                    with torch.no_grad():
                        vec = embedder(t).cpu().numpy()[0]

                    vectors.append(vec)
                    meta.append({
                        "frame": frame_idx,
                        "box": [x1, y1, x2, y2],
                        "confidence": float(prob),
                    })

                    if args.show:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if args.show:
            cv2.imshow("Faces (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  frame {frame_idx}/{total_frames} | faces saved: {len(vectors)}",
                  flush=True)

    cap.release()
    cv2.destroyAllWindows()

    if not vectors:
        print("No faces found.")
        return

    arr = np.stack(vectors)
    np.save(f"{args.out}.npy", arr)
    with open(f"{args.out}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Processed {frame_idx} frames, saved {len(vectors)} face vectors")
    print(f"  Vectors  -> {args.out}.npy  (shape {arr.shape})")
    print(f"  Metadata -> {args.out}_meta.json")


if __name__ == "__main__":
    main()