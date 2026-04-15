import os
import yaml
import json
import glob
from pathlib import Path
import mlflow
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np

from ResearchTraining.util.io import (
    load_yolo_labels,
    read_data_config,
    get_cls,
    generate_dino_labels,
    prepare_targets,
)

from ResearchTraining.util.metrics import (
    evaluate_yolo_style,
    log_yolo_metrics_to_mlflow,
)

from ResearchTraining.models.dino import run_grounding_dino

from ResearchTraining.util.metrics import evaluate_yolo_style, log_results_to_mlflow

# ---------------- CONFIG ----------------
load_dotenv()

with open("./eval_config.yaml", "r") as f:
    RUN_CONFIG = yaml.safe_load(f)

print("Configurating...")
IMAGE_DIR = RUN_CONFIG["image_dir"]
LABEL_DIR = RUN_CONFIG["label_dir"]
MODEL_ID = RUN_CONFIG["model_id"]
DATA_CONFIG = read_data_config(RUN_CONFIG["data_config"])

CLASS_NAMES = get_cls(DATA_CONFIG, clean=True)
TEXT_PROMPT = generate_dino_labels(CLASS_NAMES)

RUN_CONFIG["CLASS_NAMES"] = CLASS_NAMES
RUN_CONFIG["TEXT_PROMPT"] = TEXT_PROMPT

CONF_THRESH = RUN_CONFIG["conf_thresh"]
IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(RUN_CONFIG["experiment_name"])

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(
    f"Active Experiment: {mlflow.get_experiment_by_name(RUN_CONFIG['experiment_name'])}"
)


# ----------------- MAIN ------------------
def main():
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_paths = sorted(image_paths)
    RUN_CONFIG["valid_images"] = len(image_paths)
    print(json.dumps(RUN_CONFIG, indent=4))

    # lots of help from https://huggingface.co/docs/transformers/model_doc/grounding-dino#usage-tips
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    )

    # prediction metrics
    # metrics = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    all_preds = []
    all_targets = []

    with mlflow.start_run():
        mlflow.log_params(RUN_CONFIG)
        for img_path in tqdm(image_paths):
            img_path = Path(img_path)

            image = Image.open(img_path).convert("RGB")
            label_path = os.path.join(
                LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            )
            label = load_yolo_labels(label_path, image.width, image.height)

            pred = run_grounding_dino(
                image,
                processor,
                model,
                text_prompt=TEXT_PROMPT,
                class_to_id=CLASS_TO_ID,
            )[0]
            target = prepare_targets(label)[0]

            # metrics.update(pred, target)
            all_preds.append(pred)
            all_targets.append(target)

        # summary = evaluate_yolo_style(
        #     preds=all_preds,
        #     targets=all_targets,
        #     num_classes=len(CLASS_NAMES),
        #     iou_thresholds=np.arange(0.50, 0.96, 0.05),  # mAP50-95
        # )

        # log_yolo_metrics_to_mlflow(
        #     summary=summary,
        #     class_names=CLASS_NAMES,
        #     map_label="mAP50-95",
        # )

        summary = evaluate_yolo_style(
            preds=all_preds,
            targets=all_targets,
            num_classes=len(CLASS_NAMES),
            iou_thresholds=list(np.arange(0.5, 0.96, 0.05)),
            conf_thresholds=RUN_CONFIG["conf_thresholds"],
        )

        log_results_to_mlflow(summary, ID_TO_CLASS=ID_TO_CLASS)


if __name__ == "__main__":
    main()
