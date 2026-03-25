import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 32


def load_txt(path):
    """Load a text file into a numpy array, one row per line."""
    if not Path(path).exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(v) for v in line.split(",")])
    return np.array(rows) if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", default="app/datasets/collected")
    parser.add_argument("--original",  default="app/datasets/original",
                        help="Path to original X_train/Y_train files (optional)")
    parser.add_argument("--out",       default="app/datasets/retrain")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = ["standing","sitting","walking","using_phone","sleeping","other"]

    all_X, all_Y = [], []

    # Load newly collected data
    coll_dir = Path(args.collected)
    for action in actions:
        xf = coll_dir / f"X_{action}.txt"
        yf = coll_dir / f"Y_{action}.txt"
        if xf.exists() and yf.exists():
            X = load_txt(xf)
            Y = load_txt(yf)
            if X is not None and len(X) > 0:
                all_X.append(X)
                all_Y.append(Y)
                print(f"Collected {action}: {len(X)} rows")

    # Load original data if available
    orig_dir = Path(args.original)
    for fname in ["X_train.txt", "X_test.txt"]:
        xf = orig_dir / fname
        yf = orig_dir / fname.replace("X_", "Y_")
        if xf.exists() and yf.exists():
            X = load_txt(xf)
            Y = load_txt(yf)
            if X is not None:
                all_X.append(X)
                all_Y.append(Y)
                print(f"Original {fname}: {len(X)} rows")

    if not all_X:
        print("No data found. Run collect_data.py first.")
        return

    X = np.vstack(all_X)
    Y = np.vstack(all_Y).flatten()

    print(f"\nTotal rows: {len(X)}")

    # Reshape into windows of WINDOW_SIZE
    # Each window = WINDOW_SIZE consecutive rows with the same label
    # Group rows by label, split into windows
    X_windows, Y_windows = [], []
    for label in range(6):
        idx  = np.where(Y == label)[0]
        rows = X[idx]
        n_windows = len(rows) // WINDOW_SIZE
        for i in range(n_windows):
            window = rows[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]
            X_windows.append(window)
            Y_windows.append(label)

    X_arr = np.array(X_windows)   # (N, 32, 34)
    Y_arr = np.array(Y_windows)   # (N,)

    print(f"Total windows: {len(X_arr)}")
    for label, name in enumerate(actions):
        count = (Y_arr == label).sum()
        print(f"  {name}: {count} windows")

    # Train/test split
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_arr, Y_arr, test_size=args.test_size,
        random_state=42, stratify=Y_arr
    )

    # Save in original flat format (each window unrolled row by row)
    def save_flat(X_wins, Y_wins, x_path, y_path):
        with open(x_path, "w") as xf, open(y_path, "w") as yf:
            for win, label in zip(X_wins, Y_wins):
                for row in win:
                    xf.write(",".join(f"{v:.6f}" for v in row) + "\n")
                    yf.write(f"{int(label)}\n")

    save_flat(X_tr, Y_tr, out_dir/"X_train.txt", out_dir/"Y_train.txt")
    save_flat(X_te, Y_te, out_dir/"X_test.txt",  out_dir/"Y_test.txt")

    print(f"\nSaved to {out_dir}/")
    print(f"  Train: {len(X_tr)} windows")
    print(f"  Test:  {len(X_te)} windows")


if __name__ == "__main__":
    main()