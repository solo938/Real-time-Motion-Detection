import base64
import io
import time

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from PIL import Image

# detector_3d removed — YOLOv8-pose handles all detection now
from app.src.pose_action_recognizer import PoseActionRecognizer

analyze_bp = Blueprint("analyze", __name__, url_prefix="/analyze")

_pose_recognizer    = PoseActionRecognizer(timeout_seconds=5.0)
_live_session_start = time.time()
_person_names: dict[int, str] = {}
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

    try:
        img_bytes = base64.b64decode(data["image"])
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        _pose_recognizer.submit_frame(frame_bgr)

        persons = _pose_recognizer.get_results()
        for p in persons:
            tid = p.get("track_id")
            if tid is not None:
                p["name"] = _person_names.get(tid, f"Person {tid}")

        motion_detected, motion_regions = _pose_recognizer.get_motion()
        now = time.time()

        return jsonify({
            "persons":         persons,
            "motion_detected": motion_detected,
            "motion_regions":  motion_regions,
            "detections":      [],
            "action_events":   [],
            "session_elapsed": round(now - _live_session_start, 2),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@analyze_bp.route("/reset", methods=["POST"])
def reset_timer():
    global _live_session_start
    _pose_recognizer.reset()
    _live_session_start = time.time()
    return jsonify({"status": "reset"})


@analyze_bp.route("/active", methods=["GET"])
def active_tracks():
    now     = time.time()
    persons = _pose_recognizer.get_results()
    for p in persons:
        tid = p.get("track_id")
        if tid is not None:
            p["name"] = _person_names.get(tid, f"Person {tid}")
    motion_detected, motion_regions = _pose_recognizer.get_motion()
    return jsonify({
        "persons":         persons,
        "motion_detected": motion_detected,
        "motion_regions":  motion_regions,
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
            "motion":     p.get("motion", False),
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
