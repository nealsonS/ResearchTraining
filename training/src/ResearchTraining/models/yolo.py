from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info

import os
import yaml
import json
import glob
import mlflow
from PIL import Image

from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np

from util.io import (
    normalize_label,
    load_yolo_labels,
    read_data_config,
    get_cls,
    generate_qwen_prompt,
    parse_output_to_json,
    prepare_targets,
)

from util.metrics import evaluate_yolo_style, log_results_to_mlflow

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
