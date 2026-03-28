import argparse
import numpy as np
from pathlib import Path


MPOSE_TO_YOUR = {
    0:  2,   # walk     → walking
    1:  2,   # run      → walking
    2:  5,   # jump     → other
    3:  5,   # turn     → other
    4:  5,   # wave     → other
    5:  4,   # lie      → sleeping
    6:  1,   # sit      → sitting
    7:  0,   # stand    → standing
    8:  5,   # bend     → other
    9:  5,   # squat    → other
    10: 5,   # kick     → other
    11: 5,   # punch    → other
    12: 5,   # push     → other
    13: 5,   # pull     → other
    14: 5,   # clap     → other
    15: 5,   # throw    → other
    16: 5,   # catch    → other
    17: 5,   # climb    → other
    18: 5,   # carry    → other
    19: 3,   # phone    → using_phone
}


OPENPOSE_TO_COCO = [
    0,   # COCO 0  nose      ← OP 0
    15,  # COCO 1  l_eye     ← OP 15
    14,  # COCO 2  r_eye     ← OP 14
    17,  # COCO 3  l_ear     ← OP 17
    16,  # COCO 4  r_ear     ← OP 16
    5,   # COCO 5  l_shoulder← OP 5
    2,   # COCO 6  r_shoulder← OP 2
    6,   # COCO 7  l_elbow   ← OP 6
    3,   # COCO 8  r_elbow   ← OP 3
    7,   # COCO 9  l_wrist   ← OP 7
    4,   # COCO 10 r_wrist   ← OP 4
    11,  # COCO 11 l_hip     ← OP 11
    8,   # COCO 12 r_hip     ← OP 8
    12,  # COCO 13 l_knee    ← OP 12
    9,   # COCO 14 r_knee    ← OP 9
    13,  # COCO 15 l_ankle   ← OP 13
    10,  # COCO 16 r_ankle   ← OP 10
]

TARGET_FRAMES = 16


def remap_and_normalise(sample_frames: np.ndarray) -> np.ndarray:
    """
    sample_frames: (T, 18, 3)  OpenPose keypoints
    Returns:       (32, 34)    COCO-ordered, normalised, padded/trimmed
    """
    T = sample_frames.shape[0]

    coco = sample_frames[:, OPENPOSE_TO_COCO, :]

    normed = np.zeros((T, 17, 2), dtype=np.float32)
    for t in range(T):
        xy      = coco[t, :, :2].copy()
        conf    = coco[t, :, 2]
        visible = conf > 0.1
        if visible.sum() >= 2:
            mins = xy[visible].min(axis=0)
            maxs = xy[visible].max(axis=0)
            rng  = maxs - mins
            rng[rng < 1e-3] = 1.0
            xy = (xy - mins) / rng
        normed[t] = xy

    if T >= TARGET_FRAMES:
        start  = (T - TARGET_FRAMES) // 2
        normed = normed[start:start + TARGET_FRAMES]
    else:
        pad     = np.tile(normed[-1:], (TARGET_FRAMES - T, 1, 1))
        normed  = np.concatenate([normed, pad], axis=0)

    return normed.reshape(TARGET_FRAMES, 34)


def write_flat(X: np.ndarray, Y: np.ndarray, x_path: Path, y_path: Path):
    """
    X: (N, 32, 34)
    Y: (N,)
    Writes one row per frame, WINDOW_SIZE rows per sample.
    """
    with open(x_path, "w") as fx, open(y_path, "w") as fy:
        for i in range(len(X)):
            label = int(Y[i])
            for frame in X[i]:
                fx.write(",".join(f"{v:.6f}" for v in frame) + "\n")
                fy.write(f"{label}\n")
    print(f"  Wrote {len(X)} samples → {x_path.name}  {y_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="app/datasets/mpose",
                        help="Output directory for txt files")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3],
                        help="MPOSE2021 train/test split (1, 2 or 3)")
    parser.add_argument("--data_dir", default="./data/mpose",
                        help="Where mpose downloads raw data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MPOSE2021 split {args.split} via mpose package...")
    print(f"Raw data cache: {args.data_dir}")

    try:
        import mpose
    except ImportError:
        print("Run: pip install mpose")
        return

    dataset = mpose.MPOSE(
        pose_extractor="openpose",
        split=args.split,
        preprocess="scale_and_center",
        # data_dir removed — no longer accepted by mpose v1.2
    )
    dataset.get_info()

    X_train_raw, y_train_raw, X_test_raw, y_test_raw = dataset.get_data()

    print(f"\nRaw shapes:")
    print(f"  X_train: {X_train_raw.shape}  y_train: {y_train_raw.shape}")
    print(f"  X_test:  {X_test_raw.shape}   y_test:  {y_test_raw.shape}")
    print(f"  X dtype: {X_train_raw.dtype}  value range: [{X_train_raw.min():.2f}, {X_train_raw.max():.2f}]")

    # Process each split
    for split_name, X_raw, y_raw in [
        ("train", X_train_raw, y_train_raw),
        ("test",  X_test_raw,  y_test_raw),
    ]:
        print(f"\nProcessing {split_name}...")

        X_out = []
        Y_out = []
        skipped = 0
        label_counts = {}

        for i in range(len(X_raw)):
            mpose_label = int(y_raw[i])

            # remap to your 6 labels
            your_label = MPOSE_TO_YOUR.get(mpose_label, 5)

            sample = X_raw[i]   # (T, 18, 3) or (T, 54)

            if sample.ndim == 2:
                # flatten format — reshape to (T, 18, 3)
                T = sample.shape[0]
                n_kp = sample.shape[1] // 3
                sample = sample.reshape(T, n_kp, 3)

            if sample.shape[1] < 18:
                skipped += 1
                continue

            if sample.shape[1] > 18:
                sample = sample[:, :18, :]

            processed = remap_and_normalise(sample)  # (32, 34)
            X_out.append(processed)
            Y_out.append(your_label)
            label_counts[your_label] = label_counts.get(your_label, 0) + 1

        X_arr = np.array(X_out, dtype=np.float32)
        Y_arr = np.array(Y_out, dtype=np.int32)

        print(f"  Processed: {len(X_arr)}  skipped: {skipped}")
        label_names = {0:"standing",1:"sitting",2:"walking",3:"using_phone",4:"sleeping",5:"other"}
        for lbl, cnt in sorted(label_counts.items()):
            print(f"    {label_names[lbl]:12s} ({lbl}): {cnt}")

        write_flat(
            X_arr, Y_arr,
            out_dir / f"X_{split_name}.txt",
            out_dir / f"y_{split_name}.txt",   
        )

    print(f"\nDone. Files saved to: {out_dir}/")
    print("\nNext steps:")
    print("  python -m app.src.train --data_root app/datasets/mpose/")


if __name__ == "__main__":
    main()
