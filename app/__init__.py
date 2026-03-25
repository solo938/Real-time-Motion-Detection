from flask import Flask
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .config import config
from .main.routes import main_bp
from .upload.routes import upload_bp
from .analyze.routes import analyze_bp
from app.src.lstm import ActionClassificationLSTM
import os
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)


    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    
    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(analyze_bp, url_prefix='/analyze')

    
    @app.before_request
    def load_models():
        if not hasattr(app, 'pose_detector'):
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.DEVICE = "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
            app.pose_detector = DefaultPredictor(cfg)
            print("Detectron2 model loaded.")

        if not hasattr(app, 'lstm_classifier'):
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
            with torch.serialization.safe_globals([EarlyStopping, ModelCheckpoint]):
                app.lstm_classifier = ActionClassificationLSTM.load_from_checkpoint(
                    "app/models/saved_model.ckpt",
                    map_location = "cpu"
                    )
            app.lstm_classifier.eval()
            print("LSTM classifier loaded.")


    return app