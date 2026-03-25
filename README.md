# Real-Time Motion Detection (RTMD) 

**Real-time human action recognition web app** built with Flask, Detectron2, LSTM, and YOLOv8.

Detects and classifies **6 actions** in uploaded videos or live webcam streams:
- **Standing** | **Sitting** | **Walking** | **Using Phone** | **Sleeping** | **Other**

---

### ✨ Features

- Upload MP4 videos or use webcam
- Real-time pose estimation with Detectron2
- LSTM-based action classification (32-frame window)
- Lightweight MPOSE2021-trained model
- Clean, modern Bootstrap 5 UI
- Model retraining pipeline included
- Fully compatible with Apple Silicon (CPU / MPS)

---

### 🚀 Quick Start

#### 1. Clone the repository

```bash
git clone https://github.com/solo938/Real-time-Motion-Detection.git
cd Real-time-Motion-Detection
````

#### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set your Hugging Face token (optional)

```bash
echo "HF_TOKEN=hf_your_actual_token_here" > .env
```

#### 5. Run the application

```bash
python run.py
```

Open your browser and go to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### 📁 Project Structure

```bash
RTMD/
├── app/                  # Main Flask application
│   ├── main/             # Core routes
│   ├── upload/           # Video upload handling
│   ├── analyze/          # Analysis & inference
│   ├── src/              # Models, training, utils
│   ├── models/           # Trained checkpoints (ignored)
│   ├── templates/        # HTML templates
│   └── datasets/         # Preprocessed MPOSE data
├── static/               # CSS, JS, images
├── uploads/              # Temporary video storage
├── run.py                # Entry point
├── requirements.txt
├── .env                  # (never committed)
└── .gitignore
```

---

### 🛠️ How It Works

1. **Pose Extraction** – Detectron2 (OpenPose format)
2. **Normalization** – Scale & center + COCO keypoint remapping
3. **Action Classification** – LSTM trained on MPOSE2021
4. **Optional Tracking** – YOLOv8 + DeepSORT for multi-person scenes

---

### 🔄 Retrain the Model

```bash
# 1. Convert MPOSE dataset
python -m app.src.mpose_to_rtmd --split 1

# 2. Train new LSTM model
python -m app.src.train \
    --data_root app/datasets/mpose/ \
    --out app/models/saved_model_v2.ckpt \
    --epochs 200
```

---

### 🛡️ Security & Notes

* All large files (datasets, models, checkpoints) are excluded via `.gitignore`
* Secrets are loaded from `.env` (never committed)
* Tested and optimized for macOS Apple Silicon

---

### 🤝 Contributing

Pull requests and issues are welcome!

Ideas for improvement:

* Add more action classes
* Docker / deployment support
* UI/UX enhancements
* Real-time multi-person tracking improvements

---

### 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Flask + PyTorch**

*Star the repo if you find it useful! ⭐*

```

