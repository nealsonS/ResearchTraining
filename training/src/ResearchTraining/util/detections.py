import torch
from PIL import Image, ImageDraw
import os

# functions to modify the predictions


def resize_bounding_boxes(box: torch.Tensor, resize_by: float = 1.0):
    """
    Enlarge bounding boxes by a given factor around their centers.

    Args:
        box: Tensor of shape (N, 4) in [x1, y1, x2, y2]
        resize_by: Scale factor applied to width and height (e.g. 1.5 to grow by 50%)

    Returns:
        Scaled bounding box tensor, same shape as input
    """
    if box.numel() == 0:
        return box
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_w = (x2 - x1) / 2 * resize_by
    half_h = (y2 - y1) / 2 * resize_by
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)


def draw_pred_on_image(image_path: str, pred: dict):
    """Draw predictions on image

    Args:
        image_path (str): path to image
        pred (list[dict]): image prediction

    pred:
        {
            "boxes": FloatTensor[N, 4],
            "scores": FloatTensor[N],
            "labels": Int64Tensor[N],
        }
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image {image_path} not found. skipping...")
        im = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(im)

        boxes = pred["boxes"].tolist()
        scores = pred["scores"].tolist()
        labels = pred["labels"].tolist()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), f"{label} {score:.2f}", fill="red")

        return im
    except Exception as e:
        print(f"image {image_path} not found with error:\n{e}")
        return None
