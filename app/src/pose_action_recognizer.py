import time
import threading
import numpy as np
import torch
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

ACTION_LABELS = {0:"standing",1:"sitting",2:"walking",3:"using phone",4:"sleeping",5:"other"}

WINDOW_SIZE        = 32
MIN_VISIBLE_KP     = 2
PRED_COOLDOWN      = 4
VOTE_WINDOW        = 5
CONF_THRESHOLD     = 0.45


def _extract_keypoints(kp_tensor):
    kp      = kp_tensor.cpu().numpy()
    xy      = kp[:, :2].copy()
    visible = kp[:, 2] > 0.1
    n_vis   = int(visible.sum())

    if n_vis < MIN_VISIBLE_KP:
        return None, n_vis

    mins = xy[visible].min(axis=0)
    maxs = xy[visible].max(axis=0)
    rng  = maxs - mins
    rng[rng < 1.0] = 1.0
    xy   = (xy - mins) / rng
    return xy.flatten().astype(np.float32), n_vis


class PersonTrack:
    def __init__(self, track_id):
        self.track_id        = track_id
        self.buffer          = deque(maxlen=WINDOW_SIZE)
        self.action          = "detecting..."
        self.confidence      = 0.0
        self.start_time      = time.time()
        self.last_seen       = time.time()
        self.frames_since_pred = 0
        self.pred_history    = deque(maxlen=VOTE_WINDOW)

    def add_frame(self, kp_flat):
        self.buffer.append(kp_flat)
        self.last_seen = time.time()
        self.frames_since_pred += 1

    @property
    def ready(self):
        return len(self.buffer) == WINDOW_SIZE

    @property
    def should_predict(self):
        return self.ready and self.frames_since_pred >= PRED_COOLDOWN

    def apply_prediction(self, pred_class, conf):
        self.frames_since_pred = 0

        if conf < CONF_THRESHOLD:
            return

        self.pred_history.append((pred_class, conf))

        scores = {}
        for cls, c in self.pred_history:
            scores[cls] = scores.get(cls, 0.0) + c

        best_class = max(scores, key=scores.get)
        count = sum(1 for cls, _ in self.pred_history if cls == best_class)

        if count >= 2:
            self.action = ACTION_LABELS.get(best_class, "unknown")
            self.confidence = scores[best_class] / count

    def duration(self):
        return round(time.time() - self.start_time, 2)


def _make_deepsort():
    return DeepSort(
        max_age=30,
        n_init=1,
        max_cosine_distance=0.4,
        nn_budget=100,
        embedder="mobilenet",
        half=False,
        bgr=True,
    )


class PoseActionRecognizer:

    def __init__(self, timeout_seconds=5.0):
        self.timeout         = timeout_seconds
        self._person_tracks  = {}
        self._lock           = threading.Lock()
        self._pending_frame  = None
        self._latest_results = []
        self._busy           = False
        self._app            = None
        self._deepsort       = _make_deepsort()

        self.yolo = YOLO("yolov8s.pt")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def set_app(self, app):
        self._app = app

    def reset(self):
        with self._lock:
            self._person_tracks.clear()
            self._latest_results = []
            self._deepsort = _make_deepsort()

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._pending_frame = frame_bgr.copy()

    def get_results(self):
        with self._lock:
            return list(self._latest_results)

    def _worker(self):
        while True:
            frame = None
            with self._lock:
                if self._pending_frame is not None and not self._busy:
                    frame = self._pending_frame
                    self._pending_frame = None
                    self._busy = True

            if frame is None:
                time.sleep(0.04)
                continue

            try:
                results = self._process(frame)
                with self._lock:
                    self._latest_results = results
            except Exception as e:
                print(f"[PoseWorker] error: {e}")
            finally:
                with self._lock:
                    self._busy = False

    
    def _is_phone_in_hand(self, kp_tensor, phone_centers):
        if len(phone_centers) == 0:
            return False

        kp = kp_tensor.cpu().numpy()
        left_wrist  = kp[9][:2]
        right_wrist = kp[10][:2]

        for (px, py) in phone_centers:
            dl = np.linalg.norm(left_wrist - np.array([px, py]))
            dr = np.linalg.norm(right_wrist - np.array([px, py]))

            if dl < 80 or dr < 80:
                return True

        return False

    def _process(self, frame_bgr):
        print(f"[PROCESS] frame shape={frame_bgr.shape}")
        
        phone_centers = []
        try:
            yolo_results = self.yolo(frame_bgr, conf=0.25, verbose=False)

            for r in yolo_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = self.yolo.names[cls]

                    if name == "cell phone":
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        phone_centers.append((cx, cy))
        except Exception as e:
            print("[YOLO] error:", e)

        app = self._app
        if app is None:
            return []

        pose_detector   = getattr(app, 'pose_detector', None)
        lstm_classifier = getattr(app, 'lstm_classifier', None)

        if pose_detector is None:
            return []

        # frame_bgr is already BGR — DO NOT convert again
        outputs   = pose_detector(frame_bgr)
        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_keypoints") or len(instances) == 0:
            self._deepsort.update_tracks([], frame=frame_bgr)
            return []

        scores    = instances.scores.numpy()
        boxes     = instances.pred_boxes.tensor.numpy()
        keypoints = instances.pred_keypoints

        valid = [i for i, s in enumerate(scores) if s > 0.5]
        if not valid:
            self._deepsort.update_tracks([], frame=frame_bgr)
            return []

        raw_dets = []
        kp_map   = {}

        for idx, i in enumerate(valid):
            b = boxes[i]
            raw_dets.append(([b[0], b[1], b[2]-b[0], b[3]-b[1]], float(scores[i]), "person"))
            kp_map[idx] = keypoints[i]

        tracks = self._deepsort.update_tracks(raw_dets, frame=frame_bgr)

        now     = time.time()
        results = []

        if len(tracks) == 0:
            print("[WARNING] No confirmed tracks — using raw detections as fallback")
            for idx, i in enumerate(valid):
                kp_flat, _ = _extract_keypoints(keypoints[i])
                if kp_flat is None:
                    continue
                tid = i
                if tid not in self._person_tracks:
                    self._person_tracks[tid] = PersonTrack(tid)
                pt = self._person_tracks[tid]
                pt.add_frame(kp_flat)
                results.append({
                    "track_id":   tid,
                    "action":     pt.action,
                    "confidence": round(pt.confidence, 3),
                    "duration":   pt.duration(),
                    "bbox":       [round(float(v), 1) for v in boxes[i].tolist()],
                    "keypoints":  keypoints[i].cpu().numpy()[:, :2].tolist(),
                    "buf_size":   len(pt.buffer),
                })
            return results

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()

            tcx = (x1+x2)/2
            tcy = (y1+y2)/2

            best_idx, best_dist = None, 9999.0
            for idx, i in enumerate(valid):
                b = boxes[i]
                d = float(np.hypot(tcx-(b[0]+b[2])/2, tcy-(b[1]+b[3])/2))
                if d < best_dist:
                    best_dist, best_idx = d, idx

            if tid not in self._person_tracks:
                self._person_tracks[tid] = PersonTrack(tid)

            pt = self._person_tracks[tid]
            pt.last_seen = now

            if best_idx is not None :
                kp_flat, _ = _extract_keypoints(kp_map[best_idx])
                if kp_flat is not None:
                    pt.add_frame(kp_flat)

            
            if pt.should_predict and lstm_classifier is not None:
                x = torch.tensor(np.array(pt.buffer), dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    probs = torch.softmax(lstm_classifier(x), dim=1)
                    pred  = int(probs.argmax(dim=1).item())
                    conf  = float(probs.max().item())

                    
                    if best_idx is not None:
                        if self._is_phone_in_hand(kp_map[best_idx], phone_centers):
                            pred = 3
                            conf = max(conf, 0.95)

                pt.apply_prediction(pred, conf)

            results.append({
                "track_id":   tid,
                "action":     pt.action,
                "confidence": round(pt.confidence, 3),
                "duration":   pt.duration(),
                "bbox":       [round(float(v),1) for v in [x1,y1,x2,y2]],
                "keypoints":  kp_map[best_idx].cpu().numpy()[:,:2].tolist() if best_idx is not None else [],
                "buf_size":   len(pt.buffer),
            })

        
        stale = [tid for tid, pt in self._person_tracks.items()
                 if now - pt.last_seen > self.timeout]

        for tid in stale:
            del self._person_tracks[tid]

        return results