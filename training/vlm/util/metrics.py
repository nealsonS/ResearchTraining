import numpy as np
import torch
from torchvision.ops import box_iou
import json
import pandas as pd
import mlflow
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapezoid(y, x))


def match_predictions_to_targets(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    tgt_boxes: torch.Tensor,
    tgt_labels: torch.Tensor,
    iou_thresholds: np.ndarray,
):
    """
    Returns:
        correct: bool array [num_preds, num_iou_thresholds]
    """
    n_preds = pred_boxes.shape[0]
    n_iou = len(iou_thresholds)
    correct = np.zeros((n_preds, n_iou), dtype=bool)

    if n_preds == 0 or tgt_boxes.shape[0] == 0:
        return correct

    pred_boxes = pred_boxes.cpu()
    pred_labels = pred_labels.cpu()
    tgt_boxes = tgt_boxes.cpu()
    tgt_labels = tgt_labels.cpu()

    ious = box_iou(pred_boxes, tgt_boxes)  # [num_preds, num_targets]

    for cls in torch.unique(torch.cat([pred_labels, tgt_labels], dim=0)):
        cls = int(cls.item())
        pred_idx = torch.where(pred_labels == cls)[0]
        tgt_idx = torch.where(tgt_labels == cls)[0]

        if len(pred_idx) == 0 or len(tgt_idx) == 0:
            continue

        cls_ious = ious[pred_idx][:, tgt_idx]  # [num_cls_preds, num_cls_targets]

        # sort predictions by score desc within class
        cls_scores = pred_scores[pred_idx].cpu().numpy()
        order = np.argsort(-cls_scores)
        pred_idx = pred_idx[order]
        cls_ious = cls_ious[order]

        for t, iou_thr in enumerate(iou_thresholds):
            matched_targets = set()

            for pi in range(cls_ious.shape[0]):
                # best target for this prediction
                best_tgt_local = torch.argmax(cls_ious[pi]).item()
                best_iou = cls_ious[pi, best_tgt_local].item()
                global_tgt_idx = int(tgt_idx[best_tgt_local].item())

                if best_iou >= iou_thr and global_tgt_idx not in matched_targets:
                    correct[pred_idx[pi], t] = True
                    matched_targets.add(global_tgt_idx)

    return correct


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iou_thresholds: np.ndarray,
):
    """
    Args:
        tp: [num_preds, num_iou_thresholds] bool/int
        conf: [num_preds]
        pred_cls: [num_preds]
        target_cls: [num_targets_total]

    Returns:
        dict with:
            precision, recall, ap, f1, classes
    """
    # sort all predictions by confidence descending
    order = np.argsort(-conf)
    tp = tp[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    unique_classes = np.unique(target_cls)
    n_iou = len(iou_thresholds)

    precision = np.zeros((len(unique_classes), n_iou))
    recall = np.zeros((len(unique_classes), n_iou))
    ap = np.zeros((len(unique_classes), n_iou))

    for ci, c in enumerate(unique_classes):
        pred_mask = pred_cls == c
        n_p = pred_mask.sum()
        n_t = (target_cls == c).sum()

        if n_t == 0:
            continue
        if n_p == 0:
            continue

        tpc = tp[pred_mask].cumsum(0)
        fpc = (1 - tp[pred_mask]).cumsum(0)

        recalls = tpc / (n_t + 1e-16)
        precisions = tpc / (tpc + fpc + 1e-16)

        for j in range(n_iou):
            r = recalls[:, j]
            p = precisions[:, j]

            recall[ci, j] = r[-1] if len(r) else 0.0
            precision[ci, j] = p[-1] if len(p) else 0.0
            ap[ci, j] = compute_ap(r, p) if len(r) else 0.0

    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return {
        "classes": unique_classes,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "f1": f1,
    }


def evaluate_yolo_style(
    preds: list[dict],
    targets: list[dict],
    num_classes: int,
    iou_thresholds=None,
):
    """
    preds[i]:
        {
            "boxes": FloatTensor[N, 4],
            "scores": FloatTensor[N],
            "labels": Int64Tensor[N],
        }

    targets[i]:
        {
            "boxes": FloatTensor[M, 4],
            "labels": Int64Tensor[M],
        }
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)  # mAP50-95
    else:
        iou_thresholds = np.array(iou_thresholds, dtype=np.float32)

    stats_tp = []
    stats_conf = []
    stats_pred_cls = []
    stats_target_cls = []

    for pred, tgt in zip(preds, targets):
        pred_boxes = pred["boxes"].detach().cpu()
        pred_scores = pred["scores"].detach().cpu()
        pred_labels = pred["labels"].detach().cpu()

        tgt_boxes = tgt["boxes"].detach().cpu()
        tgt_labels = tgt["labels"].detach().cpu()

        if pred_boxes.numel() == 0:
            if tgt_labels.numel() > 0:
                stats_target_cls.append(tgt_labels.numpy())
            continue

        correct = match_predictions_to_targets(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            tgt_boxes=tgt_boxes,
            tgt_labels=tgt_labels,
            iou_thresholds=iou_thresholds,
        )

        stats_tp.append(correct.astype(np.float32))
        stats_conf.append(pred_scores.numpy())
        stats_pred_cls.append(pred_labels.numpy())

        if tgt_labels.numel() > 0:
            stats_target_cls.append(tgt_labels.numpy())

    if len(stats_target_cls) == 0:
        raise ValueError("No target labels found.")

    target_cls = np.concatenate(stats_target_cls, axis=0)

    if len(stats_tp) == 0:
        per_class = {
            "classes": np.arange(num_classes),
            "precision": np.zeros((num_classes, len(iou_thresholds))),
            "recall": np.zeros((num_classes, len(iou_thresholds))),
            "ap": np.zeros((num_classes, len(iou_thresholds))),
            "f1": np.zeros((num_classes, len(iou_thresholds))),
        }
    else:
        per_class = ap_per_class(
            tp=np.concatenate(stats_tp, axis=0),
            conf=np.concatenate(stats_conf, axis=0),
            pred_cls=np.concatenate(stats_pred_cls, axis=0),
            target_cls=target_cls,
            iou_thresholds=iou_thresholds,
        )

    # YOLO-style summary:
    # P and R usually reported at IoU=0.50 summary row
    p50 = per_class["precision"][:, 0].mean() if len(per_class["classes"]) else 0.0
    r50 = per_class["recall"][:, 0].mean() if len(per_class["classes"]) else 0.0
    map50 = per_class["ap"][:, 0].mean() if len(per_class["classes"]) else 0.0
    map_all = per_class["ap"].mean() if len(per_class["classes"]) else 0.0

    return {
        "P": float(p50),
        "R": float(r50),
        "mAP50": float(map50),
        "mAP50-95_or_custom": float(map_all),
        "iou_thresholds": iou_thresholds,
        "per_class": per_class,
    }


def log_yolo_metrics_to_mlflow(summary, class_names, map_label="mAP50-95"):
    # overall row
    mlflow.log_metrics(
        {
            "overall/P": float(summary["P"]),
            "overall/R": float(summary["R"]),
            "overall/mAP50": float(summary["mAP50"]),
            f"overall/{map_label}": float(summary["mAP50-95_or_custom"]),
        }
    )

    per_class = summary["per_class"]
    classes = per_class["classes"]

    rows = []

    for i, cls_id in enumerate(classes):
        cls_name = class_names[int(cls_id)]

        p = float(per_class["precision"][i, 0])  # IoU=0.50
        r = float(per_class["recall"][i, 0])  # IoU=0.50
        m50 = float(per_class["ap"][i, 0])  # AP@0.50
        mall = float(per_class["ap"][i].mean())  # AP averaged across IoUs

        # scalar metrics visible in MLflow metrics UI
        mlflow.log_metrics(
            {
                f"per_class/{cls_name}/P": p,
                f"per_class/{cls_name}/R": r,
                f"per_class/{cls_name}/mAP50": m50,
                f"per_class/{cls_name}/{map_label}": mall,
            }
        )

        rows.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "P": p,
                "R": r,
                "mAP50": m50,
                map_label: mall,
            }
        )

    # artifact table for easy browsing
    df = pd.DataFrame(rows)
    mlflow.log_table(df, artifact_file="metrics/per_class_metrics.json")

    # optional: also log plain dict/json
    mlflow.log_dict(rows, "metrics/per_class_metrics_list.json")


def _tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def log_predictions_to_mlflow(
    preds: list[dict],
    image_paths: list[str] | None = None,
    class_names: list[str] | None = None,
    artifact_dir: str = "predictions",
    file_name: str = "predictions.json",
):
    """
    Logs model predictions as MLflow artifacts.

    preds[i]:
        {
            "boxes": FloatTensor[N, 4],
            "scores": FloatTensor[N],
            "labels": Int64Tensor[N],
        }

    image_paths:
        Optional list of image paths aligned with preds.

    class_names:
        Optional list where class_names[label_id] -> class name.
    """
    records = []

    for i, pred in enumerate(preds):
        boxes = _tensor_to_list(pred.get("boxes", torch.empty((0, 4))))
        scores = _tensor_to_list(pred.get("scores", torch.empty((0,))))
        labels = _tensor_to_list(
            pred.get("labels", torch.empty((0,), dtype=torch.int64))
        )

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            label_id = int(label)
            label_name = (
                class_names[label_id]
                if class_names is not None and 0 <= label_id < len(class_names)
                else str(label_id)
            )

            detections.append(
                {
                    "label_id": label_id,
                    "label_name": label_name,
                    "confidence": float(score),
                    "box_xyxy": [float(x) for x in box],
                }
            )

        records.append(
            {
                "image_index": i,
                "image_path": image_paths[i] if image_paths is not None else None,
                "num_detections": len(detections),
                "detections": detections,
            }
        )

    # Log raw nested JSON
    mlflow.log_dict(records, f"{artifact_dir}/{file_name}")

    # Also log a flat table for easier browsing/filtering in MLflow UI
    flat_rows = []
    for rec in records:
        for det in rec["detections"]:
            flat_rows.append(
                {
                    "image_index": rec["image_index"],
                    "image_path": rec["image_path"],
                    "label_id": det["label_id"],
                    "label_name": det["label_name"],
                    "confidence": det["confidence"],
                    "x1": det["box_xyxy"][0],
                    "y1": det["box_xyxy"][1],
                    "x2": det["box_xyxy"][2],
                    "y2": det["box_xyxy"][3],
                }
            )

    df = pd.DataFrame(flat_rows)
    mlflow.log_table(df, f"{artifact_dir}/predictions_table.json")


def draw_and_log_predictions_to_mlflow(
    preds: list[dict],
    image_paths: list[str],
    class_names: list[str] | None = None,
    artifact_dir: str = "prediction_images",
    score_threshold: float = 0.0,
    line_width: int = 3,
):
    """
    Draw predicted boxes on images and log them as MLflow artifacts.

    preds[i]:
        {
            "boxes": FloatTensor[N, 4],   # xyxy in pixel coordinates
            "scores": FloatTensor[N],
            "labels": Int64Tensor[N],
        }

    image_paths[i]:
        path to the corresponding source image
    """
    if len(preds) != len(image_paths):
        raise ValueError(
            f"preds and image_paths must have same length, got {len(preds)} and {len(image_paths)}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (pred, image_path) in enumerate(zip(preds, image_paths)):
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            boxes = pred.get("boxes", torch.empty((0, 4))).detach().cpu()
            scores = pred.get("scores", torch.empty((0,))).detach().cpu()
            labels = (
                pred.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu()
            )

            for box, score, label in zip(boxes, scores, labels):
                score = float(score)
                if score < score_threshold:
                    continue

                x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                label_id = int(label)

                if class_names is not None and 0 <= label_id < len(class_names):
                    label_text = class_names[label_id]
                else:
                    label_text = str(label_id)

                text = f"{label_text} {score:.2f}"

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

                # Draw text background and text
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

                if font is not None:
                    bbox = draw.textbbox((x1, y1), text, font=font)
                else:
                    # fallback approximation
                    bbox = (x1, y1, x1 + 8 * len(text), y1 + 14)

                tx1, ty1, tx2, ty2 = bbox
                text_bg = [tx1, max(0, ty1 - 2), tx2 + 4, ty2 + 2]
                draw.rectangle(text_bg, fill="red")
                draw.text(
                    (x1 + 2, max(0, y1 - (ty2 - ty1) - 2)),
                    text,
                    fill="cyan",
                    font=font,
                )

            image_name = Path(image_path).stem
            output_path = os.path.join(tmpdir, f"{i:05d}_{image_name}_pred.jpg")
            image.save(output_path, quality=95)

        mlflow.log_artifacts(tmpdir, artifact_path=artifact_dir)
