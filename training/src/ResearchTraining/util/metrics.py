import torch

from torchmetrics.detection import MeanAveragePrecision

import mlflow

import pandas as pd
import numpy as np


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


def calculate_precision(validation_metrics: dict, cls: int = None) -> float:
    """Calculate precision yolo style from output dict from MeanAveragePrecision
    Calculate at:
      - IoU=0.5
      - All areas
      - 100 max detections

    If class inputted -> pick out class then calculate mean
    If class not inputted -> aggregate among all classes

    Shape is: (TxRxKxAxM)

    Args:
        validation_metrics (dict): _description_
        cls (int, optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """

    if "precision" not in validation_metrics:
        return None

    precision = validation_metrics["precision"][0, :, :, 0, -1]

    # replace -1 with NaN to avoid it messing up
    # mean values
    precision = precision.clone()  # avoid writing in-memory
    precision[precision < 0] = float("nan")

    # average among all class
    if cls is None:
        return torch.nanmean(precision).item()
    elif isinstance(cls, int):
        return torch.nanmean(precision[:, cls]).item()


def calculate_recall(validation_metrics: dict, cls: int = None) -> float:
    """Calculate recall yolo style from output dict from MeanAveragePrecision
    Calculate at:
      - IoU=0.5
      - All areas
      - 100 max detections

    If class inputted -> pick out class then calculate mean
    If class not inputted -> aggregate among all classes

    Shape is: (TxKxAxM)

    Args:
        validation_metrics (dict): _description_
        cls (int, optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """

    if "recall" not in validation_metrics:
        return None

    recall = validation_metrics["recall"][0, :, 0, -1]

    # replace -1 with NaN to avoid it messing up
    # mean values
    recall = recall.clone()  # avoid writing in-memory
    recall[recall < 0] = float("nan")

    # average among all class
    if cls is None:
        return torch.nanmean(recall).item()
    elif isinstance(cls, int):
        return torch.nanmean(recall[cls]).item()


# TODO- handle case when label is in the YAML file
# but is not in labels
def evaluate_yolo_style(
    preds: list[dict],
    targets: list[dict],
    num_classes: int,
    iou_thresholds=list[int],
    conf_thresholds=list[float],
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

    YOLO calculates precision and recall basically average precision/recall at IoU=0.5 and single confidence threshold
    mAP50 basically is average precision/recall at ALL confidence threshold
    """
    output = {}
    for conf in conf_thresholds:
        # calculate scalar precision and recall first
        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5],
            class_metrics=True,
            extended_summary=True,
        )
        pred_conf_filtered = filter_preds_by_score(preds=preds, conf_threshold=conf)
        metric.update(pred_conf_filtered, targets)
        val_met = metric.compute()

        # calculate map50
        # no filter for conf threshold
        metric_map50 = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5],
            class_metrics=True,
            # extended_summary=True,
        )

        metric_map50.update(preds, targets)
        map50 = metric_map50.compute()

        metric_map50_95 = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=iou_thresholds,
            class_metrics=True,
            # extended_summary=True,
        )

        metric_map50_95.update(preds, targets)
        map50_95 = metric_map50_95.compute()

        # get actual classes that appear in the labels
        present_classes = torch.cat([t["labels"] for t in targets]).unique().tolist()
        present_classes = [int(c) for c in present_classes]

        results = [
            {
                "class": cls,
                "precision": calculate_precision(val_met, cls),
                "recall": calculate_recall(val_met, cls),
                "map_50": map50["map_per_class"][cls].item(),
                "map_50_95": map50_95["map_per_class"][cls].item(),
            }
            for cls in present_classes
        ] + [
            {
                "class": -1,
                "precision": calculate_precision(val_met),
                "recall": calculate_recall(val_met),
                "map_50": map50["map"].item(),
                "map_50_95": map50_95["map"].item(),
            }
        ]

        output[conf] = results

    return output


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
                        "overall/conf_thresh": float(conf),
                        "overall/P": float(result["precision"]),
                        "overall/R": float(result["recall"]),
                        "overall/mAP50": float(result["map_50"]),
                        "overall/mAP50-95": float(result["map_50_95"]),
                    }
                )
            else:
                cls_name = ID_TO_CLASS.get(result["class"], "unk")
                mlflow.log_metrics(
                    {
                        f"per_class/{cls_name}/conf_thresh": float(conf),
                        f"per_class/{cls_name}/P": float(result["precision"]),
                        f"per_class/{cls_name}/R": float(result["recall"]),
                        f"per_class/{cls_name}/mAP50": float(result["map_50"]),
                        f"per_class/{cls_name}/mAP50-95": float(result["map_50_95"]),
                    }
                )

            result["class_name"] = ID_TO_CLASS.get(result["class"], "unk")
            result["confidence_threshold"] = conf
            df_rows.append(result)

        df = pd.DataFrame(df_rows)

        mlflow.log_table(
            df, artifact_file=f"metrics/per_class_metrics_conf_{conf}.json"
        )
