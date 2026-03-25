import base64
import io
import time

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from PIL import Image

from app.src.detector_3d import run_on_frame
from app.src.action_timer import ActionTimer
from app.src.pose_action_recognizer import PoseActionRecognizer

analyze_bp = Blueprint("analyze", __name__, url_prefix="/analyze")

_live_timer         = ActionTimer(timeout_seconds=2.0)
_pose_recognizer    = PoseActionRecognizer(timeout_seconds=5.0)
_live_session_start = time.time()
_person_names: dict[int, str] = {}


class MotionDetector:

    def __init__(self, threshold=20, min_area=400):
        self.threshold = threshold
        self.min_area  = min_area
        self._prev     = None

    def reset(self):
        self._prev = None

    def detect(self, pil_img, persons: list[dict]) -> list[dict]:
        frame = np.array(pil_img)
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray  = cv2.GaussianBlur(gray, (15, 15), 0)

        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray
            return []

        diff        = cv2.absdiff(self._prev, gray)
        self._prev  = gray
        _, thresh   = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh      = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = int(x + w // 2), int(y + h // 2)

            # link to nearest person
            nearest_id, nearest_dist = None, 9999.0
            for p in persons:
                bbox = p.get("bbox", [])
                if len(bbox) == 4:
                    pcx  = (bbox[0] + bbox[2]) / 2
                    pcy  = (bbox[1] + bbox[3]) / 2
                    dist = float(np.hypot(cx - pcx, cy - pcy))
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_id   = p.get("track_id")

            regions.append({
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "area": int(area),
                "cx": cx, "cy": cy,
                "person_id": nearest_id,
            })
        return regions


_motion    = MotionDetector()
_app_wired = False


@analyze_bp.before_app_request
def _wire_app():
    global _app_wired
    if not _app_wired:
        _pose_recognizer.set_app(current_app._get_current_object())
        _app_wired = True


@analyze_bp.route("/frame", methods=["POST"])
def analyze_frame():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    threshold = float(data.get("threshold", 0.65))

    try:
        img_bytes = base64.b64decode(data["image"])
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        detections, _ = run_on_frame(pil_img, threshold=threshold)

        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        _pose_recognizer.submit_frame(frame_bgr)

        persons = _pose_recognizer.get_results()
        for p in persons:
            tid = p.get("track_id")
            p["name"] = _person_names.get(tid, f"Person {tid}")

        motion_regions = _motion.detect(pil_img, persons=persons)

        now           = time.time()
        action_events = _live_timer.update(detections, now=now)

        return jsonify({
            "detections":      [d.to_dict() for d in detections],
            "action_events":   action_events,
            "persons":         persons,
            "motion_regions":  motion_regions,
            "session_elapsed": round(now - _live_session_start, 2),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@analyze_bp.route("/reset", methods=["POST"])
def reset_timer():
    global _live_session_start
    _live_timer.reset()
    _pose_recognizer.reset()
    _motion.reset()
    _live_session_start = time.time()
    return jsonify({"status": "timer reset"})


@analyze_bp.route("/active", methods=["GET"])
def active_tracks():
    now     = time.time()
    persons = _pose_recognizer.get_results()
    for p in persons:
        tid    = p.get("track_id")
        p["name"] = _person_names.get(tid, f"Person {tid}")
    return jsonify({
        "active":          _live_timer.get_active(now=now),
        "persons":         persons,
        "session_elapsed": round(now - _live_session_start, 2),
    })


@analyze_bp.route("/people", methods=["GET"])
def known_people():
    persons = _pose_recognizer.get_results()
    people  = []
    for p in persons:
        tid = p.get("track_id")
        people.append({
            "track_id":   tid,
            "name":       _person_names.get(tid, f"Person {tid}"),
            "action":     p.get("action", "detecting..."),
            "confidence": p.get("confidence", 0),
            "duration":   p.get("duration", 0),
        })
    return jsonify({"people": people})


@analyze_bp.route("/people/rename", methods=["POST"])
def rename_person():
    data = request.get_json(force=True)
    tid  = data.get("track_id")
    name = data.get("name", "").strip()
    if tid is None or not name:
        return jsonify({"error": "track_id and name required"}), 400
    _person_names[int(tid)] = name
    return jsonify({"status": "renamed", "track_id": tid, "name": name})