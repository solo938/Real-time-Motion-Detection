import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
 
from PIL import Image
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_SIZE = 518
CMAP = plt.cm.get_cmap("tab20", 91)
 

_detr_processor = None
_detr_model = None
_depth_processor = None
_depth_model = None
 
 
def _load_models():
    global _detr_processor, _detr_model, _depth_processor, _depth_model
    if _detr_model is not None:
        return  
 
    _detr_processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    _detr_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    ).to(DEVICE)
    _detr_model.eval()
 
    _depth_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    _depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(DEVICE)
    _depth_model.eval()
 
 

 
def get_color(label_id: int) -> tuple:
    rgba = CMAP(int(label_id) % 91)
    return tuple(int(x * 255) for x in rgba[:3])
 
 
def _run_detection(pil_img: Image.Image, threshold: float = 0.65):
    inputs = _detr_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _detr_model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = _detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]
    return (
        results["boxes"].cpu().numpy(),
        results["labels"].cpu().numpy(),
        results["scores"].cpu().numpy(),
    )
 
 
def _run_depth(pil_img: Image.Image) -> np.ndarray:
    small = pil_img.resize((DEPTH_SIZE, DEPTH_SIZE))
    inputs = _depth_processor(images=small, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        raw = _depth_model(**inputs).predicted_depth
    depth = raw.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth  
 
 
def _get_box_depth(depth_map: np.ndarray, box, orig_w: int, orig_h: int) -> float:
    dh, dw = depth_map.shape
    sx, sy = dw / orig_w, dh / orig_h
    x1 = max(0, int(box[0] * sx))
    y1 = max(0, int(box[1] * sy))
    x2 = min(dw, int(box[2] * sx))
    y2 = min(dh, int(box[3] * sy))
    region = depth_map[y1:y2, x1:x2]
    return float(region.mean()) if region.size > 0 else 0.5
 
 
def depth_colormap(depth_np: np.ndarray) -> Image.Image:
    colored = (cm.magma(depth_np)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)
 
 

 
class Detection3DResult:
    
 
    def __init__(self, label_id, label_name, score, box, depth_z, extrusion, color):
        self.label_id = int(label_id)
        self.label_name = str(label_name)
        self.score = float(score)
        self.box = [float(v) for v in box]
        self.depth_z = float(depth_z)
        self.extrusion = int(extrusion)
        self.color = color
 
    def to_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "label_name": self.label_name,
            "score": round(self.score, 4),
            "box": [round(v, 1) for v in self.box],
            "depth_z": round(self.depth_z, 4),
            "extrusion": self.extrusion,
            "color": self.color,
        }
 
 
def run_on_frame(
    pil_img: Image.Image,
    threshold: float = 0.65,
    max_size: int = 640,
) -> tuple[list[Detection3DResult], np.ndarray]:

    _load_models()

    orig_w, orig_h = pil_img.size

    # Resize for speed but keep scale factors to map boxes back
    scale = min(1.0, max_size / max(orig_w, orig_h))
    if scale < 1.0:
        resized = pil_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
    else:
        resized = pil_img

    resized_w, resized_h = resized.size
    boxes, labels, scores = _run_detection(resized, threshold=threshold)
    depth_map = _run_depth(resized)

    # Scale boxes back to original image size
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        # box is in resized coords — scale back to original
        orig_box = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y,
        ]
        z = _get_box_depth(depth_map, box, resized_w, resized_h)
        ext = max(5, int(20 * (1.0 - z)))
        col = get_color(label)
        name = _detr_model.config.id2label.get(int(label), str(label))
        detections.append(
            Detection3DResult(
                label_id=label,
                label_name=name,
                score=score,
                box=orig_box,
                depth_z=z,
                extrusion=ext,
                color=col,
            )
        )

    return detections, depth_map