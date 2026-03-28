import time
import threading
import numpy as np
import torch
import cv2
from collections import deque
from queue import Queue
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

DEVICE = "cpu"
print(f"[PoseActionRecognizer] Strategy: Hybrid Async Pipeline (Device: {DEVICE.upper()})")

ACTION_LABELS = {0:"standing",1:"sitting",2:"walking",3:"using phone",4:"sleeping",5:"other"}
WINDOW_SIZE    = 16
PRED_COOLDOWN  = 2
VOTE_WINDOW    = 5      
CONF_THRESHOLD = 0.40
MIN_VISIBLE_KP = 4      

MOTION_SCALE     = (320, 240)
MOTION_BLUR      = (11, 11)
MOTION_THRESH_PX = 25
MOTION_MIN_AREA  = 30


def _extract_keypoints(kp):
    if isinstance(kp, torch.Tensor):
        kp = kp.cpu().numpy()
    if kp.ndim == 3:
        kp = kp[0]
    xy      = kp[:, :2].copy()
    visible = kp[:, 2] > 0.3   
    n_vis   = int(visible.sum())
    if n_vis < MIN_VISIBLE_KP:
        return None, n_vis
    mins, maxs = xy[visible].min(axis=0), xy[visible].max(axis=0)
    rng = maxs - mins
    rng[rng < 1.0] = 1.0
    xy = (xy - mins) / rng
    return xy.flatten().astype(np.float32), n_vis


def _classify_pose(kp, bbox):
    """
    Geometry-based pose classifier.
    Returns (action_str, confidence_float).

    Key insight — pixel y increases downward:
      STANDING: knee_y > hip_y  (knees below hips)
      SITTING:  knee_y < hip_y  (knees above/same level as hips when legs horizontal)
      SLEEPING: nose_y ≈ hip_y  (body is horizontal)
    """
    x1, y1, x2, y2 = bbox
    bh = max(y2 - y1, 1)
    bw = max(x2 - x1, 1)

    nose_y     = kp[0,  1]
    l_sho_y    = kp[5,  1]
    r_sho_y    = kp[6,  1]
    l_hip_y    = kp[11, 1]
    r_hip_y    = kp[12, 1]
    l_kne_y    = kp[13, 1]
    r_kne_y    = kp[14, 1]
    l_ankle_y  = kp[15, 1]
    r_ankle_y  = kp[16, 1]

    shoulder_y = (l_sho_y + r_sho_y) / 2
    hip_y      = (l_hip_y + r_hip_y) / 2
    knee_y     = (l_kne_y + r_kne_y) / 2
    ankle_y    = (l_ankle_y + r_ankle_y) / 2 

    
    hip_vis  = (kp[11, 2] + kp[12, 2]) / 2
    knee_vis = (kp[13, 2] + kp[14, 2]) / 2
    ankle_vis= (kp[15, 2] + kp[16, 2]) / 2


    if hip_vis > 0.3 and ankle_vis > 0.3:
        leg_length_ratio = abs(hip_y - ankle_y) / bh
        if leg_length_ratio > 0.55:
                return "standing", 0.90

   
    nose_hip_diff = abs(nose_y - hip_y) / bh
    if nose_hip_diff < 0.15:
        return "sleeping", 0.85
    
    if hip_vis > 0.3 and knee_vis > 0.3:
        if knee_y < hip_y:  
            return "sitting", 0.80
        

    box_w = x2 - x1
    aspect_ratio = bh / max(box_w, 1)
    if aspect_ratio < 1.6:
        return "sitting", 0.70
    
    return "standing", 0.60

   


class PersonTrack:
    def __init__(self, track_id):
        self.track_id          = track_id
        self.buffer            = deque(maxlen=WINDOW_SIZE)
        self.action            = "detecting..."
        self.confidence        = 0.0
        self.start_time        = time.time()
        self.last_seen         = time.time()
        self.frames_since_pred = 0
        self.pred_history      = deque(maxlen=VOTE_WINDOW)
        self.last_bbox         = None
        
        self.geo_action        = "detecting..."
        self.geo_conf          = 0.0
        self.phone_grace_counter      = 0
        self.final_history = deque(maxlen=7)

    def add_frame(self, kp_flat):
        self.buffer.append(kp_flat)
        self.frames_since_pred += 1

    @property
    def should_predict(self):
        return len(self.buffer) == WINDOW_SIZE and self.frames_since_pred >= PRED_COOLDOWN

    def apply_prediction(self, pred_class, conf):
        self.frames_since_pred = 0
        if conf < CONF_THRESHOLD:
            return
        self.pred_history.append((pred_class, conf))
        scores = {}
        for cls, _ in self.pred_history:
            scores[cls] = scores.get(cls, 0.0) + 1
        best_class = max(scores, key=scores.get)
        
        if scores[best_class] >= 4:  
            self.action     = ACTION_LABELS.get(best_class, "unknown")
            self.confidence = conf

    def best_action(self):
        """
        Return the most reliable action:
        - If LSTM has voted confidently → use LSTM
        - Otherwise → use geometry (instant, no warmup needed)
        """

        if self.geo_action == "using phone":
            return "using phone", self.geo_conf


        if self.geo_action == "walking" and self.geo_conf > 0.85:
            return "walking", self.geo_conf
        if self.confidence >= 0.75 and self.action not in ("detecting...", "other"):
            return self.action, self.confidence
        if self.geo_action != "detecting...":
            return self.geo_action, self.geo_conf
        return "detecting...", 0.0

    def duration(self):
        return round(time.time() - self.start_time, 2)


class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def analyze(self, frame_bgr, person_tracks):
        sw, sh = MOTION_SCALE
        small  = cv2.resize(frame_bgr, (sw, sh))
        gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray   = cv2.GaussianBlur(gray, MOTION_BLUR, 0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return False, []

        diff        = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thresh   = cv2.threshold(diff, MOTION_THRESH_PX, 255, cv2.THRESH_BINARY)
        thresh      = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        orig_h, orig_w = frame_bgr.shape[:2]
        sx = orig_w / sw
        sy = orig_h / sh
        regions         = []
        motion_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MOTION_MIN_AREA:
                continue
            motion_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            ox, oy = int(x*sx), int(y*sy)
            ow, oh = int(w*sx), int(h*sy)
            ocx, ocy = ox+ow//2, oy+oh//2

            nearest_id, nearest_dist = None, 9999.0
            for tid, pt in person_tracks.items():
                if pt.last_bbox is None:
                    continue
                bx1, by1, bx2, by2 = pt.last_bbox
                d = float(np.hypot(ocx-(bx1+bx2)/2, ocy-(by1+by2)/2))
                if d < nearest_dist:
                    nearest_dist, nearest_id = d, tid

            regions.append({
                "x": ox, "y": oy, "w": ow, "h": oh,
                "area": int(area*sx*sy),
                "cx": ocx, "cy": ocy,
                "person_id": str(nearest_id) if nearest_id else None,
            })

        return motion_detected, regions


class PoseActionRecognizer:
    def __init__(self, timeout_seconds=5.0):
        self.timeout        = timeout_seconds
        self._person_tracks = {}
        self._lock          = threading.Lock()
        self._pending_frame = None
        self._latest_results= []
        self._busy          = False
        self._app           = None
        self._motion        = MotionAnalyzer()

        self.pose_model = YOLO("yolov8n-pose.pt")
        self.yolo       = YOLO("yolov8n.pt")
        self._deepsort  = DeepSort(
            max_age=20,
            n_init=2,               # confirm faster
            max_cosine_distance=0.3,
            nn_budget=100,
        )
        self.lstm_queue = Queue(maxsize=20)

        threading.Thread(target=self._worker,      daemon=True).start()
        threading.Thread(target=self._lstm_worker, daemon=True).start()

    def set_app(self, app):
        self._app = app

    def reset(self):
        with self._lock:
            self._person_tracks.clear()
            self._latest_results = []
            self._motion.reset()
            self._deepsort = DeepSort(max_age=20, n_init=2, max_cosine_distance=0.3)

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._pending_frame = frame_bgr.copy()

    def get_results(self):
        with self._lock:
            return list(self._latest_results)

    def get_motion(self):
        return (
            getattr(self, '_last_motion_detected', False),
            getattr(self, '_last_motion_regions',  []),
        )

    def _lstm_worker(self):
        while True:
            track, seq_np = self.lstm_queue.get()
            try:
                if self._app and getattr(self._app, 'lstm_classifier', None):
                    t = torch.from_numpy(seq_np).unsqueeze(0).float().to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(self._app.lstm_classifier(t), dim=1)
                        conf, pred = torch.max(probs, dim=1)
                    track.apply_prediction(int(pred.item()), float(conf.item()))
            except Exception as e:
                print(f"[LSTM Worker] {e}")
            finally:
                self.lstm_queue.task_done()

    def _is_phone_in_hand(self, kp, phone_centers, bbox):
        if not phone_centers:
            return False

        x1, y1, x2, y2 = bbox
        scale = max(x2 - x1, y2 - y1)

    
        thresh = 0.25 * scale   

        lw = kp[9, :2]
        rw = kp[10, :2]

        for (px, py) in phone_centers:
            if np.linalg.norm(lw - [px, py]) < thresh:
               return True
            if np.linalg.norm(rw - [px, py]) < thresh:
                return True

        return False

    def _worker(self):
        while True:
            frame = None
            with self._lock:
                if self._pending_frame is not None and not self._busy:
                    frame, self._pending_frame = self._pending_frame, None
                    self._busy = True
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                res = self._process(frame)
                with self._lock:
                    self._latest_results = res
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[PoseWorker] {e}")
            finally:
                with self._lock:
                    self._busy = False

    def _process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        
        motion_detected, motion_regions = self._motion.analyze(
            frame_bgr, self._person_tracks)

        
        phone_centers = []
        try:
            for r in self.yolo(frame_bgr, conf=0.15, verbose=False):
                for b in r.boxes:
                    if self.yolo.names[int(b.cls[0])] == "cell phone":
                        c = b.xyxy[0].cpu().numpy()
                        phone_centers.append(((c[0]+c[2])/2, (c[1]+c[3])/2))
        except Exception as e:
            print(f"[YOLO] {e}")

       
        pose_res = self.pose_model(frame_bgr, conf=0.5, verbose=False)[0]  # raised to 0.5
        dets, kp_list = [], []

        if pose_res.boxes is not None and pose_res.keypoints is not None:
            for i, b in enumerate(pose_res.boxes):
                
                bx1,by1,bx2,by2 = b.xyxy[0].cpu().numpy()
                bbox_w = bx2 - bx1
                bbox_h = by2 - by1
                if bbox_w > w*0.95 or bbox_h > h*0.95:
                    continue   
                kp = pose_res.keypoints.data[i].cpu().numpy()
                vis = (kp[:, 2] > 0.3).sum()
                if vis < MIN_VISIBLE_KP:
                    continue   

                dets.append(([float(bx1),float(by1),float(bbox_w),float(bbox_h)],
                              float(b.conf[0]), "person"))
                kp_list.append(kp)

        if not dets:
            self._last_motion_detected = motion_detected
            self._last_motion_regions  = motion_regions
            return []

        
        tracks  = self._deepsort.update_tracks(dets, frame=frame_bgr)
        now     = time.time()
        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()
            tx1,ty1,tx2,ty2 = ltrb

            
            if (tx2-tx1) > w*0.95 or (ty2-ty1) > h*0.95:
                continue

            tcx = (tx1+tx2)/2; tcy = (ty1+ty2)/2

            
            best_kp, best_dist = None, 9999.0
            for i, kp in enumerate(kp_list):
                bx1 = dets[i][0][0]; bx2 = bx1+dets[i][0][2]
                by1 = dets[i][0][1]; by2 = by1+dets[i][0][3]
                d = float(np.hypot(tcx-(bx1+bx2)/2, tcy-(by1+by2)/2))
                if d < best_dist:
                    best_dist, best_kp = d, kp

            if best_kp is None or best_dist > 200:
                continue

            pt = self._person_tracks.setdefault(tid, PersonTrack(tid))
            pt.last_bbox = (tx1, ty1, tx2, ty2)
            pt.last_seen = now

            
            is_phone = self._is_phone_in_hand(best_kp, phone_centers, (tx1,ty1,tx2,ty2))
            kp_flat, n_vis = _extract_keypoints(best_kp)
            if is_phone:
                pt.phone_grace_counter = 30
                pt.geo_action = "using phone"
                pt.geo_conf   = 0.95
                pt.final_history.clear()
                pt.final_history.extend(["using phone"] * 5)
            
            elif pt.phone_grace_counter > 0:
                pt.phone_grace_counter -= 1
                pt.geo_action = "using phone"
                pt.geo_conf   = 0.90

            if pt.geo_action == "using phone":
                pt.buffer.clear()  
                

            else:
                mv = 0.0
                if kp_flat is not None and  n_vis > 8 and len(pt.buffer) >= 2:
                    mv = float(np.linalg.norm(
                        np.array(kp_flat) - np.array(pt.buffer[-1])))

                if mv > 0.8:
                    pt.geo_action = "walking"
                    pt.geo_conf   = 0.90
                else:
                    pt.geo_action, pt.geo_conf = _classify_pose(
                        best_kp, (tx1,ty1,tx2,ty2))

            
            if kp_flat is not None:
                kp_norm = kp_flat.copy()

                kp_norm = kp_norm - np.mean(kp_norm)
                kp_norm = kp_norm / (np.std(kp_norm) + 1e-6)
                pt.add_frame(kp_norm)
                if pt.should_predict and not self.lstm_queue.full():
                    self.lstm_queue.put((pt, np.array(list(pt.buffer))))

            
            final_action, final_conf = pt.best_action()


            if pt.geo_action == "using phone" or pt.phone_grace_counter > 0:
                final_action = "using phone"
                final_conf   = 0.95
                pt.final_history.clear()
                pt.final_history.extend(["using phone"] * 7)

            else:
   
               pt.final_history.append(final_action)
               final_action = max(set(pt.final_history), key=pt.final_history.count)

            person_moving = any(r["person_id"] == str(tid) for r in motion_regions)
            print(
    f"[POSE] id={tid} "
    f"geo={pt.geo_action}({pt.geo_conf:.2f}) "
    f"lstm={pt.action}({pt.confidence:.2f}) "
    f"FINAL={final_action}"
)
            results.append({
                "track_id":   str(tid),
                "action":     final_action,
                "confidence": round(final_conf, 3),
                "duration":   pt.duration(),
                "bbox":       [float(tx1),float(ty1),float(tx2),float(ty2)],
                "keypoints":  best_kp[:, :2].tolist(),
                "motion":     person_moving,
                "buf_size":   len(pt.buffer),
            })

        # Expire stale tracks
        stale = [tid for tid, pt in self._person_tracks.items()
                 if now - getattr(pt,'last_seen',now) > self.timeout]
        for tid in stale:
            del self._person_tracks[tid]

        self._last_motion_detected = motion_detected
        self._last_motion_regions  = motion_regions
        return results
