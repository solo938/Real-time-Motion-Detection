from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class TrackedDetection:
    """
    Mirror of Detection3DResult fields plus a stable track_id.
    All original fields are forwarded so the rest of the pipeline
    (draw_pseudo3d_cv2, ActionTimer) can use them unchanged.
    """
    track_id:   int
    label_name: str
    score:      float
    box:        tuple          # (x1, y1, x2, y2)
    depth_z:    float
    color:      tuple          # (r, g, b)
    extrusion:  int

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "track_id":   self.track_id,
            "label_name": self.label_name,
            "score":      round(self.score, 4),
            "box":        [round(float(v), 1) for v in [x1, y1, x2, y2]],
            "depth_z":    round(self.depth_z, 4),
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class DeepSortObjectTracker:
    

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
        embedder: str = "mobilenet",
        center_distance_threshold: float = 150.0,
    ):
        self._ds = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            half=False,
            bgr=True,                   # VideoAnalyzer passes BGR frames
        )
        self._dist_thresh = center_distance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections,          # list[Detection3DResult] from detector_3d
        frame_bgr: np.ndarray,
    ) -> list[TrackedDetection]:
        """
        Run one tracking step.

        Returns a list of TrackedDetection — one per *confirmed* track.
        Detections that do not match any confirmed track are silently dropped
        (they will reappear once DeepSORT promotes them after n_init frames).
        """
        if not detections:
            self._ds.update_tracks([], frame=frame_bgr)
            return []

        # 1. Build the raw-detection list DeepSORT expects:
        #    ([x, y, w, h], confidence, class_label)
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            raw_dets.append((
                [x1, y1, x2 - x1, y2 - y1],
                float(det.score),
                det.label_name,
            ))

        # 2. Run DeepSORT
        tracks = self._ds.update_tracks(raw_dets, frame=frame_bgr)

        # 3. Pre-compute detection centres for nearest-neighbour matching
        det_centres = np.array(
            [self._centre(d.box) for d in detections], dtype=np.float32
        )

        # 4. For every confirmed track find the closest original detection
        #    so we can carry its metadata (depth_z, color, extrusion, score).
        results: list[TrackedDetection] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb   = track.to_ltrb()
            tx1, ty1, tx2, ty2 = ltrb
            track_cx = (tx1 + tx2) / 2.0
            track_cy = (ty1 + ty2) / 2.0

            best_idx, best_dist = self._nearest(
                track_cx, track_cy, det_centres
            )

            if best_idx is None or best_dist > self._dist_thresh:
                # Track exists but no close raw detection this frame —
                # use track bbox and fall back to sensible defaults.
                results.append(TrackedDetection(
                    track_id=track.track_id,
                    label_name=track.det_class or "unknown",
                    score=0.0,
                    box=(tx1, ty1, tx2, ty2),
                    depth_z=0.5,
                    color=(200, 200, 200),
                    extrusion=8,
                ))
            else:
                src = detections[best_idx]
                results.append(TrackedDetection(
                    track_id=track.track_id,
                    label_name=src.label_name,
                    score=src.score,
                    box=(tx1, ty1, tx2, ty2),   # use smoothed track bbox
                    depth_z=src.depth_z,
                    color=src.color,
                    extrusion=src.extrusion,
                ))

        return results

    def reset(self) -> None:
        """Re-initialise the tracker (call between videos)."""
        self._ds = DeepSort(
            max_age=self._ds.max_age,
            n_init=self._ds.n_init,
            max_cosine_distance=self._ds.max_cosine_distance,
            nn_budget=self._ds.nn_budget,
            embedder="mobilenet",
            half=False,
            bgr=True,
        )

   
    @staticmethod
    def _centre(box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _nearest(
        cx: float,
        cy: float,
        centres: np.ndarray,
    ) -> tuple[Optional[int], float]:
        if len(centres) == 0:
            return None, float("inf")
        diffs = centres - np.array([cx, cy], dtype=np.float32)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        idx   = int(dists.argmin())
        return idx, float(dists[idx])