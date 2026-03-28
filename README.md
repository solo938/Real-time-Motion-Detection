# Real-Time Motion Detection & Action Recognition

## Overview

This project implements a **real-time human action recognition pipeline** using:

* YOLOv8 (pose + object detection)
* DeepSORT (multi-object tracking)
* Geometry-based classification
* Temporal modeling (buffer + optional LSTM)
* Motion detection (frame differencing)

The system processes video frames and outputs **per-person actions with tracking IDs**.

---

## System Architecture

### Pipeline

```
Input Frame
    ↓
YOLOv8 Pose Detection
    ↓
Keypoint Extraction
    ↓
DeepSORT Tracking (ID assignment)
    ↓
Motion Analyzer (frame differencing)
    ↓
Action Recognition
    ├── Geometry-based classifier
    ├── Temporal buffer (deque)
    └── LSTM (async inference)
    ↓
Decision Layer (voting + overrides)
    ↓
Output (JSON)
```

---

### Components

#### 1. Pose Detection

* Model: `yolov8n-pose.pt`
* Outputs:

  * Bounding boxes
  * Keypoints (x, y, visibility)

#### 2. Object Detection (Phone)

* Model: `yolov8n.pt`
* Used for detecting `"cell phone"` class

#### 3. Tracking

* Library: DeepSORT
* Maintains consistent `track_id`
* Handles occlusion and re-identification

#### 4. Motion Analyzer

* Frame resizing + grayscale + Gaussian blur
* Frame differencing (`absdiff`)
* Thresholding + contour extraction
* Maps motion regions to tracked persons

#### 5. Action Recognition

##### Geometry-Based

* Uses relative joint positions:

  * Standing → legs extended
  * Sitting → knees above hips
  * Sleeping → horizontal alignment

##### Motion-Based

* Uses displacement between consecutive frames
* Detects walking vs static states

##### Phone Detection

* Wrist proximity to detected phone
* Adaptive threshold (bbox-scaled)
* Temporal persistence via grace counter

##### Temporal Modeling

* Sliding window buffer (`deque`)
* Optional LSTM inference (async thread)

#### 6. Decision Layer

* Combines:

  * Geometry prediction
  * LSTM prediction
* Applies:

  * Confidence thresholding
  * Voting (history buffer)
  * Hard overrides (e.g., phone usage)

---

## Threading Model

* **Worker Thread**

  * Processes incoming frames
  * Runs detection, tracking, classification

* **LSTM Thread**

  * Consumes sequences from queue
  * Runs model inference asynchronously

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/solo938/Real-time-Motion-Detection.git
cd Real-time-Motion-Detection
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements file is missing:

```bash
pip install opencv-python numpy torch ultralytics deep-sort-realtime
```

---

### 3. Model Weights

* YOLO models are downloaded automatically by `ultralytics`
* Ensure internet connection on first run

---

## Usage

### Run Server

```bash
python app.py
```

Server runs at:

```
http://127.0.0.1:8080
```

---

### API Endpoint

#### POST `/analyze/frame`

Send an image frame:

```bash
curl -X POST http://127.0.0.1:8080/analyze/frame \
-F "file=@frame.jpg"
```

---

### Response Format

```json
[
  {
    "track_id": "1",
    "action": "walking",
    "confidence": 0.91,
    "bbox": [x1, y1, x2, y2],
    "keypoints": [[x, y], ...],
    "motion": true
  }
]
```

---

## Configuration

Key parameters (in code):

* `WINDOW_SIZE` → sequence length for temporal modeling
* `CONF_THRESHOLD` → minimum confidence for LSTM
* `MOTION_THRESH_PX` → motion detection sensitivity
* `MIN_VISIBLE_KP` → keypoint quality filter

---

## Notes

* Designed for CPU usage (GPU optional)
* Real-time performance depends on input resolution
* Works best with:

  * Stable lighting
  * Clear human visibility

---

## Limitations

* Phone detection depends on YOLO accuracy
* Fast motion may reduce keypoint stability
* Heavy occlusion affects tracking consistency

---

## License

MIT License

