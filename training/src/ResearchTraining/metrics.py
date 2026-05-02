import torch
from collections import defaultdict

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

import mlflow

import pandas as pd


def filter_preds_by_score(preds: list[dict], conf_threshold: float) -> list[dict]:
    """preds[i]:
    {
        "boxes": FloatTensor[N, 4],
        "scores": FloatTensor[N],
        "labels": Int64Tensor[N],
    }
    """
    output = []
    for pred in preds:
        boxes, scores, labels = pred.values()
        mask = scores > conf_threshold

        boxes = boxes[mask, :]
        scores = scores[mask]
        labels = labels[mask]

        if len(boxes) == 0:
            output.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            )
        else:
            output.append({"boxes": boxes, "scores": scores, "labels": labels})
    return output


def _match_predictions(
    preds: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
) -> dict[int, tuple[int, int, int]]:
    """Match predictions to ground truth at a fixed IoU threshold.

    Predictions should already be confidence-filtered before calling this.
    Matching is greedy: predictions are processed in descending confidence
    order (matching YOLO's approach); each GT box can only be matched once.

    Returns:
        dict mapping class_id → (TP, FP, n_gt)
    """
    tp_per_class: dict[int, int] = defaultdict(int)
    fp_per_class: dict[int, int] = defaultdict(int)
    n_gt_per_class: dict[int, int] = defaultdict(int)

    for tgt in targets:
        for lbl in tgt["labels"].tolist():
            n_gt_per_class[int(lbl)] += 1

    for pred, tgt in zip(preds, targets):
        pred_boxes = pred["boxes"]  # (N, 4)
        pred_scores = pred["scores"]  # (N,)
        pred_labels = pred["labels"]  # (N,)
        gt_boxes = tgt["boxes"]  # (M, 4)
        gt_labels = tgt["labels"]  # (M,)

        if len(pred_boxes) == 0:
            continue

        # Sort by confidence descending (matches YOLO's greedy assignment)
        sort_idx = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_labels = pred_labels[sort_idx]

        if len(gt_boxes) == 0:
            for lbl in pred_labels.tolist():
                fp_per_class[int(lbl)] += 1
            continue

        iou = box_iou(pred_boxes, gt_boxes)  # (N, M)
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i in range(len(pred_boxes)):
            pred_lbl = int(pred_labels[i].item())
            same_class = (gt_labels == pred_labels[i]) & ~gt_matched

            if not same_class.any():
                fp_per_class[pred_lbl] += 1
                continue

            row_iou = iou[i].clone()
            row_iou[~same_class] = 0.0
            best_iou, best_j = row_iou.max(dim=0)

            if best_iou.item() >= iou_threshold:
                tp_per_class[pred_lbl] += 1
                gt_matched[best_j] = True
            else:
                fp_per_class[pred_lbl] += 1

    all_classes = set(n_gt_per_class) | set(fp_per_class) | set(tp_per_class)
    return {
        cls: (tp_per_class[cls], fp_per_class[cls], n_gt_per_class[cls])
        for cls in all_classes
    }


def evaluate_yolo_style(
    preds: list[dict],
    targets: list[dict],
    iou_thresholds: list[float],
    conf_thresholds: list[float],
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

    Precision and recall are computed YOLO-style:
      - Filter predictions at each confidence threshold
      - Greedy TP/FP matching at IoU=0.5 (highest-confidence predictions matched first)
      - P = TP / (TP + FP),  R = TP / n_gt  per class; overall = mean across classes

    mAP50 and mAP50-95 are confidence-threshold independent and computed once
    over all predictions (AUC of precision-recall curve).
    """
    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
    targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

    # mAP is independent of confidence threshold — compute once
    metric_map50 = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=True,
    )
    metric_map50.update(preds, targets)
    map50 = metric_map50.compute()

    metric_map50_95 = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        class_metrics=True,
    )
    metric_map50_95.update(preds, targets)
    map50_95 = metric_map50_95.compute()

    # map_per_class is indexed by position, not class ID — build lookup
    map50_cls_to_idx = {int(c): i for i, c in enumerate(map50["classes"].tolist())}
    map50_95_cls_to_idx = {
        int(c): i for i, c in enumerate(map50_95["classes"].tolist())
    }

    present_classes = torch.cat([t["labels"] for t in targets]).unique().tolist()
    present_classes = [int(c) for c in present_classes]

    output = {}
    for conf in conf_thresholds:
        pred_conf_filtered = filter_preds_by_score(preds=preds, conf_threshold=conf)
        match_results = _match_predictions(pred_conf_filtered, targets)

        results = []
        for cls in present_classes:
            tp, fp, n_gt = match_results.get(cls, (0, 0, 0))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / n_gt if n_gt > 0 else 0.0
            results.append(
                {
                    "class": cls,
                    "precision": p,
                    "recall": r,
                    "map_50": map50["map_per_class"][map50_cls_to_idx[cls]].item(),
                    "map_50_95": map50_95["map_per_class"][
                        map50_95_cls_to_idx[cls]
                    ].item(),
                }
            )

        # Overall: macro-average P/R across classes (matches YOLO's mp/mr)
        overall_p = (
            sum(r["precision"] for r in results) / len(results) if results else 0.0
        )
        overall_r = sum(r["recall"] for r in results) / len(results) if results else 0.0
        results.append(
            {
                "class": -1,
                "precision": overall_p,
                "recall": overall_r,
                "map_50": map50["map"].item(),
                "map_50_95": map50_95["map"].item(),
            }
        )

        output[conf] = results

    return output


def log_predictions_to_mlflow(
    image_paths: list[str],
    preds: list[dict],
    targets: list[dict],
    id_to_class: dict[int, str],
) -> None:
    rows = []
    for img_path, pred, target in zip(image_paths, preds, targets):
        for box, score, label in zip(
            pred["boxes"].tolist(),
            pred["scores"].tolist(),
            pred["labels"].tolist(),
        ):
            rows.append(
                {
                    "image": img_path,
                    "type": "pred",
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "score": score,
                    "label": id_to_class.get(label, label),
                }
            )
        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            rows.append(
                {
                    "image": img_path,
                    "type": "gt",
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "score": None,
                    "label": id_to_class.get(label, label),
                }
            )
    mlflow.log_table(data=pd.DataFrame(rows), artifact_file="predictions.csv")


def log_results_to_mlflow(
    summary: dict[float, dict], ID_TO_CLASS: dict[int, str]
) -> None:
    """Log Everything to MLFlow

    1. log each metric by itself
    2. write results as a csv file to Bytes

    Args:
        results[conf_thresh][i] = {
          "class": int,
          "conf_thresh": int,
          "precision": float,
          "recall": float,
          "map_50": float,
          "map_50_95": float
        }
    """

    ID_TO_CLASS[-1] = "overall"
    for conf, results in summary.items():
        df_rows = []

        for result in results:
            if result["class"] == -1:
                mlflow.log_metrics(
                    {
                        f"overall/conf_{conf}/P": float(result["precision"]),
                        f"overall/conf_{conf}/R": float(result["recall"]),
                        f"overall/conf_{conf}/mAP50": float(result["map_50"]),
                        f"overall/conf_{conf}/mAP50-95": float(result["map_50_95"]),
                    }
                )
            else:
                cls_name = ID_TO_CLASS.get(result["class"], "unk")
                mlflow.log_metrics(
                    {
                        f"per_class/{cls_name}/conf_{conf}/P": float(
                            result["precision"]
                        ),
                        f"per_class/{cls_name}/conf_{conf}/R": float(result["recall"]),
                        f"per_class/{cls_name}/conf_{conf}/mAP50": float(
                            result["map_50"]
                        ),
                        f"per_class/{cls_name}/conf_{conf}/mAP50-95": float(
                            result["map_50_95"]
                        ),
                    }
                )

            result["class_name"] = ID_TO_CLASS.get(result["class"], "unk")
            result["confidence_threshold"] = conf
            df_rows.append(result)

        df = pd.DataFrame(df_rows)

        mlflow.log_table(
            df, artifact_file=f"metrics/per_class_metrics_conf_{conf}.json"
        )
