import cv2
import time
import numpy as np
from PIL import Image
from typing import Generator
 
from app.src.detector_3d import run_on_frame, depth_colormap
from app.src.action_timer import ActionTimer
from app.src.deep_sort_tracker import DeepSortObjectTracker, TrackedDetection
 
 

 
def draw_pseudo3d_cv2(
    frame_bgr: np.ndarray,
    detections,
    action_events: list[dict],
) -> np.ndarray:
    
    timing = {e["label"]: e["duration"] for e in action_events if e["state"] == "active"}
 
    img = frame_bgr.copy()
 
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        ext = det.extrusion
        r, g, b = det.color
        front_col = (b, g, r)           
        back_col  = tuple(max(0, c - 80) for c in (b, g, r))
 
        # Back rect
        cv2.rectangle(img, (x1 - ext, y1 - ext), (x2 - ext, y2 - ext), back_col, 1)
        # Depth edge lines
        for (cx, cy, dx, dy) in [
            (x1, y1, x1 - ext, y1 - ext),
            (x2, y1, x2 - ext, y1 - ext),
            (x2, y2, x2 - ext, y2 - ext),
            (x1, y2, x1 - ext, y2 - ext),
        ]:
            cv2.line(img, (cx, cy), (dx, dy), front_col, 1)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), front_col, 2)
 
        
        dur = timing.get(det.label_name)
        dur_str = f" {dur:.1f}s" if dur is not None else ""
        tid_str = f"#{det.track_id}" if hasattr(det, 'track_id') else ""
        label = f" {tid_str}{det.label_name} {det.score:.2f} Z:{det.depth_z:.2f}{dur_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (b, g, r), -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
 
    return img
 
 

 
class VideoAnalyzer:
    
 
    def __init__(
        self,
        threshold: float = 0.65,
        frame_skip: int = 1,
        timeout_seconds: float = 2.0,
    ):
        self.threshold = threshold
        self.frame_skip = frame_skip
        self.timer = ActionTimer(timeout_seconds=timeout_seconds)
        self.tracker = DeepSortObjectTracker(max_age=30, n_init=3)
 
    def analyze_video(self, video_path: str) -> Generator[dict, None, None]:
       
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
 
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.timer.reset()
        self.tracker.reset()
        frame_index = 0
 
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
 
                if frame_index % self.frame_skip != 0:
                    frame_index += 1
                    continue
 
                timestamp_sec = frame_index / fps
                pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
 
                detections, depth_map = run_on_frame(pil_img, threshold=self.threshold)
                detections = self.tracker.update(detections, frame_bgr=frame_bgr)
                action_events = self.timer.update(detections, now=timestamp_sec)
                annotated = draw_pseudo3d_cv2(frame_bgr, detections, action_events)
 
                yield {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "detections": [d.to_dict() for d in detections],
                    "action_events": action_events,
                    "annotated_frame": annotated,
                }
 
                frame_index += 1
        finally:
            cap.release()
 
    def analyze_summary(self, video_path: str) -> dict:
        
        label_durations: dict[str, list[float]] = {}
        all_detections: list[dict] = []
        frame_count = 0
 
        for result in self.analyze_video(video_path):
            frame_count += 1
            all_detections.extend(result["detections"])
            for event in result["action_events"]:
                lbl = event["label"]
                if event["state"] == "active":
                    label_durations.setdefault(lbl, [])
                    label_durations[lbl].append(event["duration"])
 
        
        action_summary = {
            lbl: {"max_duration_sec": round(max(durs), 2), "frame_appearances": len(durs)}
            for lbl, durs in label_durations.items()
        }
 
        return {
            "frames_processed": frame_count,
            "total_detections": len(all_detections),
            "unique_labels": sorted(action_summary.keys()),
            "action_summary": action_summary,
        }