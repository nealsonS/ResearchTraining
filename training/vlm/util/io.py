import os
import yaml


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


def generate_dino_labels(labels: list[dict], id_to_name: dict[int:str]) -> str:
    """For each label, generate the text prompt for each label"""
    text_prompt = ""

    for label in labels:
        if "class_id" not in label:
            continue

        cls = label["class_id"]
        if cls not in id_to_name:
            continue

        cls_name = id_to_name[cls]
        text_prompt += f"{cls_name}. "

    return text_prompt.strip()
