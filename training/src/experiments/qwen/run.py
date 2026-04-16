from transformers import AutoModelForImageTextToText, AutoProcessor
import torch


import os
import yaml
import json
import mlflow

from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np

from ResearchTraining.util.io import (
    read_data_config,
    get_cls,
    prepare_targets,
    get_images_from_dir,
    get_label_from_image,
)

from ResearchTraining.metrics import evaluate_yolo_style, log_results_to_mlflow

from ResearchTraining.models.qwen import run_qwen_inference, generate_qwen_prompt

# ---------------- CONFIG ----------------
load_dotenv()
torch.manual_seed(1234)

with open("./eval_config.yaml", "r") as f:
    RUN_CONFIG = yaml.safe_load(f)

print("Configurating...")
IMAGE_DIR = RUN_CONFIG["image_dir"]
LABEL_DIR = RUN_CONFIG["label_dir"]
MODEL_ID = RUN_CONFIG["model_id"]
DATA_CONFIG = read_data_config(RUN_CONFIG["data_config"])

CLASS_NAMES = get_cls(DATA_CONFIG, clean=True)
TEXT_PROMPT = generate_qwen_prompt(CLASS_NAMES)

RUN_CONFIG["CLASS_NAMES"] = CLASS_NAMES
RUN_CONFIG["TEXT_PROMPT"] = TEXT_PROMPT

CONF_THRESHOLDS = RUN_CONFIG["conf_thresholds"]
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


def main():
    """Main Function

    1. Load dataset
    2. Load model and processor
    3. Generate output per image with text prompt
    4. Parse output to JSON
    5. Calculate metrics
    """
    image_paths = get_images_from_dir(IMAGE_DIR)
    RUN_CONFIG["valid_images"] = len(image_paths)

    print(json.dumps(RUN_CONFIG, indent=4))

    model = AutoModelForImageTextToText.from_pretrained(
        RUN_CONFIG["model_id"],
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=DEVICE,
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(RUN_CONFIG["model_id"])

    all_preds, all_targets = [], []
    with mlflow.start_run():
        mlflow.log_params(RUN_CONFIG)
        for img_path in tqdm(image_paths):
            label = get_label_from_image(img_path, LABEL_DIR)
            pred = run_qwen_inference(
                img_path,
                processor,
                model,
                text_prompt=TEXT_PROMPT,
                class_to_id=CLASS_TO_ID,
            )[0]
            target = prepare_targets(label)[0]

            all_preds.append(pred)
            all_targets.append(target)

        summary = evaluate_yolo_style(
            preds=all_preds,
            targets=all_targets,
            iou_thresholds=list(np.arange(0.5, 0.96, 0.05)),
            conf_thresholds=RUN_CONFIG["conf_thresholds"],
        )

        log_results_to_mlflow(summary, ID_TO_CLASS=ID_TO_CLASS)


if __name__ == "__main__":
    main()
