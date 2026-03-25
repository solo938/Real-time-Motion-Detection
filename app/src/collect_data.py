import os
import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


ACTION_TO_LABEL = {
    "standing":    0,
    "sitting":     1,
    "walking":     2,
    "using_phone": 3,
    "sleeping":    4,
    "other":       5,
}

WINDOW_SIZE = 32
MIN_VIS_KP  = 4


def load_detector():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)


def extract_keypoints(kp_tensor):
    """Returns flat 34-float array or None if not enough visible keypoints."""
    kp      = kp_tensor.cpu().numpy()   
    xy      = kp[:, :2].copy()
    visible = kp[:, 2] > 0.3
    if visible.sum() < MIN_VIS_KP:
        return None
    mins = xy[visible].min(axis=0)
    maxs = xy[visible].max(axis=0)
    rng  = maxs - mins
    rng[rng < 1.0] = 1.0
    xy = (xy - mins) / rng
    return xy.flatten()


def draw_skeleton(frame, kp_np):
    """Draw keypoints and skeleton on frame for visual feedback."""
    BONES = [
        (15,13),(13,11),(16,14),(14,12),(11,12),
        (5,11),(6,12),(5,6),(5,7),(6,8),(7,9),(8,10),
        (1,2),(0,1),(0,2),(1,3),(2,4)
    ]
    h, w = frame.shape[:2]
    pts = []
    for i in range(17):
        x, y, c = kp_np[i]
        px, py = int(x), int(y)
        pts.append((px, py, c))
        if c > 0.3:
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
    for a, b in BONES:
        if pts[a][2] > 0.3 and pts[b][2] > 0.3:
            cv2.line(frame, (pts[a][0], pts[a][1]),
                     (pts[b][0], pts[b][1]), (0, 200, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True,
                        choices=list(ACTION_TO_LABEL.keys()),
                        help="Action class to record")
    parser.add_argument("--out", default="app/datasets/collected",
                        help="Output directory")
    parser.add_argument("--target", type=int, default=300,
                        help="Target number of windows to collect (default 300)")
    args = parser.parse_args()

    label     = ACTION_TO_LABEL[args.action]
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_file = out_dir / f"X_{args.action}.txt"
    y_file = out_dir / f"Y_{args.action}.txt"

    print(f"\nLoading Detectron2...")
    detector = load_detector()
    print(f"Ready. Recording action: '{args.action}' (label={label})")
    print(f"Target: {args.target} windows × {WINDOW_SIZE} frames")
    print(f"Output: {x_file}")
    print(f"\nControls:  SPACE = start/stop recording   Q = quit & save\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording      = False
    frame_buffer   = deque(maxlen=WINDOW_SIZE)  
    windows_saved  = 0
    total_frames   = 0
    skipped_frames = 0

    x_rows = []   
    y_rows = []   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Run Detectron2
        outputs   = detector(frame)
        instances = outputs["instances"].to("cpu")

        best_kp  = None
        best_kp_raw = None

        if instances.has("pred_keypoints") and len(instances) > 0:
            # pick highest-score detection
            scores = instances.scores.numpy()
            best_i = int(scores.argmax())
            if scores[best_i] > 0.7:
                kp      = instances.pred_keypoints[best_i]
                kp_np   = kp.cpu().numpy()
                kp_flat = extract_keypoints(kp)
                if kp_flat is not None:
                    best_kp     = kp_flat
                    best_kp_raw = kp_np
                    display     = draw_skeleton(display, kp_np)

        # Collect frame if recording and keypoints are good
        if recording and best_kp is not None:
            frame_buffer.append(best_kp)
            total_frames += 1
            if len(frame_buffer) == WINDOW_SIZE:
                # save one window
                for row in frame_buffer:
                    x_rows.append(",".join(f"{v:.6f}" for v in row))
                    y_rows.append(str(label))
                windows_saved += 1
                
                for _ in range(8):
                    if frame_buffer:
                        frame_buffer.popleft()
        elif recording and best_kp is None:
            skipped_frames += 1

        # HUD overlay
        status_col = (0, 255, 0) if recording else (0, 0, 255)
        status_txt = "RECORDING" if recording else "PAUSED"
        cv2.rectangle(display, (0, 0), (520, 80), (0, 0, 0), -1)
        cv2.putText(display, f"{status_txt} | action: {args.action}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)
        cv2.putText(display,
                    f"windows: {windows_saved}/{args.target}  "
                    f"frames: {total_frames}  skipped: {skipped_frames}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        
        pct = min(1.0, windows_saved / args.target)
        cv2.rectangle(display, (0, 710), (int(1280 * pct), 720), (0, 255, 100), -1)

        cv2.imshow(f"Collecting: {args.action}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
            frame_buffer.clear()
            print(f"{'▶ Recording' if recording else '⏸ Paused'}")
        elif key == ord('q') or windows_saved >= args.target:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not x_rows:
        print("No data collected. Exiting.")
        return

    
    with open(x_file, "w") as f:
        f.write("\n".join(x_rows) + "\n")
    with open(y_file, "w") as f:
        f.write("\n".join(y_rows) + "\n")

    print(f"\nSaved {windows_saved} windows ({len(x_rows)} rows) to:")
    print(f"  {x_file}")
    print(f"  {y_file}")


if __name__ == "__main__":
    main()