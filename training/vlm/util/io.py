import os
import yaml
import json
import torch


# -------------- CONFIG READINGS --------------
def read_data_config(data_config: str):
    if not os.path.exists(data_config):
        return []

    with open(data_config, "r") as f:
        DATA_CONFIG = yaml.safe_load(f)

    return DATA_CONFIG


def get_cls(data_config: dict, clean=False) -> list[str]:
    classes = data_config["names"]

    if clean:
        classes = [cls.replace("-", " ") for cls in classes.values()]
    return classes


# -------------- BOX HELPERS --------------
def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Convert xyxy bbox label to yolo
    Returns:
        list[int]: bbox in [x1,y1,x2,y2]
    """
    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def xyxy_to_yolo(box, img_w, img_h):
    """Convert xyxy bbox label to yolo format
    XYXY: x_min, y_min, x_max, y_max
    YOLO: <class_id> <x_center> <y_center> <width> <height>

    Args:
        line (str): _description_
    """
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [xc, yc, w, h]


# ------------- IO / PARSING --------------
def load_yolo_labels(label_path, img_w, img_h):
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            cls = int(float(cls))
            xc, yc, w, h = map(float, (xc, yc, w, h))
            gts.append(
                {"class_id": cls, "box": yolo_to_xyxy(xc, yc, w, h, img_w, img_h)}
            )
    return gts


def normalize_label(label):
    label = label.strip().lower()
    return label


def parse_output_to_json(output: str):
    try:
        boxes = json.loads(output.strip())
        return boxes

    except Exception:
        return []


def prepare_targets(label: list[dict]):
    if len(label) == 0:
        return [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]

    return [
        {
            "boxes": torch.tensor([line["box"] for line in label], dtype=torch.float32),
            "labels": torch.tensor(
                [line["class_id"] for line in label], dtype=torch.int64
            ),
        }
    ]


# ------------- Label Generation --------------
def generate_dino_labels(classes: list[str]) -> str:
    """For each label, generate the text prompt for each label"""
    text_prompt = ""

    for cls in classes:
        text_prompt += f"{cls}. "

    return text_prompt.strip()


def generate_qwen_prompt(classes: list[str]) -> str:
    return f"""
You are an object detection model.

Detect all instances of the following classes in the image:
{classes}

Return ONLY a valid JSON array.

Each element must be:
[class_name, confidence, [x1, y1, x2, y2]]

Rules:
- Coordinates must be integers
- Coordinates are in pixel space
- (x1, y1) is top-left, (x2, y2) is bottom-right
- confidence is a float between 0 and 1
- Do not include any explanations or text outside JSON
- If no objects are found, return []

Example:
[
  ["amazon smile logo", 0.8, [12, 12, 30, 30]],
  ["usps logo", 0.7, [50, 100, 70, 120]]
]
"""
