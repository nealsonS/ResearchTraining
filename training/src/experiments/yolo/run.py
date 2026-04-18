from dotenv import load_dotenv
import torch
from ultralytics import YOLO, settings
import mlflow
import os
import numpy as np

import yaml

from ResearchTraining.util.io import (
    read_data_config,
    get_cls,
    get_images_from_dir,
    get_label_from_image,
    prepare_targets,
)
from ResearchTraining.models.yolo import (
    train_yolo,
    log_yolo_mlflow,
    run_yolo_batch_inference,
)

from ResearchTraining.metrics import (
    evaluate_yolo_style,
    log_results_to_mlflow,
    log_predictions_to_mlflow,
)

# ---------------- CONFIG ----------------
load_dotenv()
torch.manual_seed(1234)

with open("./config.yaml", "r") as f:
    RUN_CONFIG = yaml.safe_load(f)

print("Configurating...")
IMAGE_DIR = RUN_CONFIG["valid_images"]
LABEL_DIR = RUN_CONFIG["valid_labels"]
DATA_CONFIG = read_data_config(RUN_CONFIG["data_config"])

CLASS_NAMES = get_cls(DATA_CONFIG, clean=True)

RUN_CONFIG["CLASS_NAMES"] = CLASS_NAMES

CONF_THRESHOLDS = RUN_CONFIG["conf_thresholds"]
IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(RUN_CONFIG["MLFLOW_EXPERIMENT_NAME"])

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(
    f"Active Experiment: {mlflow.get_experiment_by_name(RUN_CONFIG['MLFLOW_EXPERIMENT_NAME'])}"
)

# unstable
settings.update({"mlflow": False})


def main():

    yolo = YOLO(RUN_CONFIG["YOLO"]["model_id"])

    with mlflow.start_run(
        run_name=(
            RUN_CONFIG["MLFLOW_RUN"] if RUN_CONFIG["MLFLOW_RUN"] is not None else None
        )
    ):
        image_paths = get_images_from_dir(IMAGE_DIR)
        RUN_CONFIG["valid_images_count"] = len(image_paths)

        mlflow.log_params(RUN_CONFIG)
        if RUN_CONFIG["YOLO"]["train"]:
            yolo, train_results = train_yolo(
                yolo,
                RUN_CONFIG["data_config"],
                RUN_CONFIG["YOLO"]["epochs"],
                RUN_CONFIG["YOLO"]["imgsz"],
                RUN_CONFIG["YOLO"]["batch_size"],
            )
            print(train_results)

        if RUN_CONFIG["YOLO"]["save_model"]:
            log_yolo_mlflow(yolo)

        if RUN_CONFIG["YOLO"]["eval"]:
            labels = [
                get_label_from_image(img_path, LABEL_DIR) for img_path in image_paths
            ]
            all_targets = [prepare_targets(label)[0] for label in labels]
            all_preds = run_yolo_batch_inference(image_paths, yolo)

            assert len(all_targets) == len(all_preds)

        if RUN_CONFIG["log_predictions"]:
            log_predictions_to_mlflow(image_paths, all_preds, all_targets, ID_TO_CLASS)

            summary = evaluate_yolo_style(
                preds=all_preds,
                targets=all_targets,
                iou_thresholds=list(np.arange(0.5, 0.96, 0.05)),
                conf_thresholds=RUN_CONFIG["conf_thresholds"],
            )
            log_results_to_mlflow(summary, ID_TO_CLASS=ID_TO_CLASS)


if __name__ == "__main__":
    main()
