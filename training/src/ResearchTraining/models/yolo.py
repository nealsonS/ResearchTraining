from pathlib import Path
from ultralytics import YOLO
import torch
import mlflow


def train_yolo(
    model: YOLO, data_yaml_path: str, epochs: int, imgsz: int, batch_size: int
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(data_yaml_path).exists():
        raise FileNotFoundError(f"{data_yaml_path} not found")

    train_results = model.train(
        data=data_yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size, device=device
    )

    return model, train_results


def run_yolo_inference(image_path: str, model: YOLO):
    results = model(image_path)
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy.numel() == 0:
        return [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]

    return [
        {
            "boxes": boxes.xyxy.to(dtype=torch.float32),
            "scores": boxes.conf.to(dtype=torch.float32),
            "labels": boxes.cls.to(dtype=torch.int64),
        }
    ]


def run_yolo_batch_inference(image_paths: list[str], model: YOLO):
    results = model(image_paths)
    all_preds = []
    for r in results:
        boxes = r.boxes
        if boxes is None or boxes.xyxy.numel() == 0:
            pred = [
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            ]
        else:
            pred = [
                {
                    "boxes": boxes.xyxy.to(dtype=torch.float32),
                    "scores": boxes.conf.to(dtype=torch.float32),
                    "labels": boxes.cls.to(dtype=torch.int64),
                }
            ]
        all_preds.extend(pred)
    return all_preds


def log_yolo_mlflow(model: YOLO):
    mlflow.log_artifact(model.trainier.best)
