import torch

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
