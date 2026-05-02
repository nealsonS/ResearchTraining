import torch

# functions to modify the predictions


def resize_bounding_boxes(box: torch.Tensor, resize_by: int = 1):
    """
    Scale bounding box coordinates by a given factor.

    Args:
        box: Tensor of shape (..., 4) in [x1, y1, x2, y2]
        resize_by: Scale factor (e.g. 0.5 to halve, 2.0 to double)

    Returns:
        Scaled bounding box tensor
    """
    return box * resize_by
