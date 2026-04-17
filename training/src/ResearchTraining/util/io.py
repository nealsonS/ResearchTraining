import os
import yaml
import json
import torch
import glob
from PIL import Image
from pathlib import Path


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
def get_images_from_dir(img_dir: str):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    image_paths = sorted(image_paths)
    return image_paths


def load_yolo_labels(label_path, img_w, img_h, convert_xyxy=True) -> list[dict]:
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
            if convert_xyxy:
                gts.append(
                    {"class_id": cls, "box": yolo_to_xyxy(xc, yc, w, h, img_w, img_h)}
                )
            else:
                gts.append({"class_id": cls, "box": [xc, yc, w, h]})
    return gts


def get_label_path_from_image(img_path: str, label_dir: str) -> str:
    return os.path.join(
        label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )


def get_label_from_image(
    img_path: str, label_dir: str, convert_xyxy=True
) -> list[dict]:
    if not Path(img_path).exists():
        raise FileNotFoundError(f"image {img_path} not found!")
    if not Path(label_dir).exists():
        raise FileNotFoundError(f"label directory {label_dir} not found!")

    image = Image.open(img_path).convert("RGB")
    label_path = get_label_path_from_image(img_path, label_dir)
    label = load_yolo_labels(
        label_path, image.width, image.height, convert_xyxy=convert_xyxy
    )
    return label


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


def write_labels_to_file(labels: dict, dest: str):
    if Path(dest).exists:
        raise FileExistsError(f"file {dest} exists!")

    lines = [
        f"{label['class_id']} {' '.join(label['box'])}".strip() for label in labels
    ]
    lines_str = "\n".join(lines)
    with open(dest, "w") as f:
        f.write(lines_str)
