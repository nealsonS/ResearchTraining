from contextlib import contextmanager
from dotenv import load_dotenv
import torch
from PIL import Image
from pathlib import Path

from ultralytics import YOLO, settings
import mlflow
import os
import numpy as np
import shutil
import tempfile

from transformers import AutoModelForImageTextToText, AutoProcessor

import yaml

from ResearchTraining.util.io import (
    read_data_config,
    get_cls,
    get_images_from_dir,
    get_label_from_image,
    prepare_targets,
    get_label_path_from_image,
    write_labels_to_file,
)

from ResearchTraining.models.yolo import (
    train_yolo,
    log_yolo_mlflow,
    run_yolo_batch_inference,
)

from ResearchTraining.models.qwen import (
    generate_classification_prompt,
    run_qwen_classification_inference,
)

from ResearchTraining.metrics import evaluate_yolo_style, log_results_to_mlflow

# ---------------- CONFIG ----------------
load_dotenv()
torch.manual_seed(1234)

with open("./config.yaml", "r") as f:
    RUN_CONFIG = yaml.safe_load(f)

print("Configurating...")
TRAIN_IMAGES_DIR = RUN_CONFIG["train_images"]
TRAIN_LABELS_DIR = RUN_CONFIG["train_labels"]

VALID_IMAGES_DIR = RUN_CONFIG["valid_images"]
VALID_LABELS_DIR = RUN_CONFIG["valid_labels"]

DATA_CONFIG = read_data_config(RUN_CONFIG["data_config"])

CLASS_NAMES = get_cls(DATA_CONFIG, clean=True)

RUN_CONFIG["CLASS_NAMES"] = CLASS_NAMES

CONF_THRESHOLDS = RUN_CONFIG["conf_thresholds"]
IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_CONFIG["classification_prompt"] = generate_classification_prompt(CLASS_NAMES)

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


@contextmanager
def single_class_labels(labels_dir: str, image_paths: list, data_config: str):
    original_data = yaml.safe_load(Path(data_config).read_text())
    single_class_data = {**original_data, "names": {0: "object"}, "nc": 1}

    backup = Path(labels_dir).parent / "_labels_backup"
    shutil.move(labels_dir, backup)
    Path(labels_dir).mkdir()

    tmp_yaml = Path(data_config).parent / "_single_class_data.yaml"
    tmp_yaml.write_text(yaml.dump(single_class_data))

    try:
        for img_path in image_paths:
            label_path = Path(get_label_path_from_image(img_path, str(backup)))
            labels = get_label_from_image(img_path, str(backup), convert_xyxy=False)
            single_class = [{"class_id": 0, "box": lbl["box"]} for lbl in labels]
            write_labels_to_file(single_class, Path(labels_dir) / label_path.name)
        yield str(tmp_yaml)
    finally:
        shutil.rmtree(labels_dir)
        shutil.move(str(backup), labels_dir)
        tmp_yaml.unlink(missing_ok=True)


def main():
    yolo = YOLO(RUN_CONFIG["YOLO"]["model_id"])
    model = AutoModelForImageTextToText.from_pretrained(
        RUN_CONFIG["QWEN"]["model_id"],
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=DEVICE,
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(RUN_CONFIG["QWEN"]["model_id"])

    image_paths = get_images_from_dir(TRAIN_IMAGES_DIR)

    RUN_CONFIG["single_class_train"] = True

    with single_class_labels(TRAIN_LABELS_DIR, image_paths, RUN_CONFIG["data_config"]) as single_class_data_config, mlflow.start_run(
        run_name=RUN_CONFIG["MLFLOW_RUN"] or None
    ):
        mlflow.log_params(RUN_CONFIG)
        if RUN_CONFIG["YOLO"]["train"]:
            yolo, train_results = train_yolo(
                yolo,
                single_class_data_config,
                RUN_CONFIG["YOLO"]["epochs"],
                RUN_CONFIG["YOLO"]["imgsz"],
                RUN_CONFIG["YOLO"]["batch_size"],
            )
            print(train_results)

        if RUN_CONFIG["YOLO"]["save_model"]:
            log_yolo_mlflow(yolo)

        if RUN_CONFIG["YOLO"]["eval"]:
            valid_images = get_images_from_dir(VALID_IMAGES_DIR)

            labels = [
                get_label_from_image(img_path, VALID_LABELS_DIR, convert_xyxy=True)
                for img_path in valid_images
            ]
            all_targets = [prepare_targets(label)[0] for label in labels]
            all_preds = run_yolo_batch_inference(valid_images, yolo)

            assert len(all_targets) == len(all_preds)

            # crop image and then ask Qwen to classify
            final_preds = []
            for val_img_path, pred in zip(valid_images, all_preds):
                image = Image.open(val_img_path).convert("RGB")
                scores = []
                pred_labels = []
                for box in pred["boxes"].tolist():
                    cropped_image = image.crop(box)
                    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                        cropped_image.save(tmp)
                        qwen_pred = run_qwen_classification_inference(
                            tmp,
                            processor,
                            model,
                            RUN_CONFIG["classification_prompt"],
                            class_to_id=CLASS_TO_ID,
                        )[0]
                    if qwen_pred["scores"].numel() == 0:
                        continue
                    best = qwen_pred["scores"].argmax()
                    scores.append(qwen_pred["scores"][best].item())
                    pred_labels.append(qwen_pred["labels"][best].item())

                final_preds.append(
                    {
                        "boxes": pred["boxes"],
                        "scores": torch.tensor(scores, dtype=torch.float32),
                        "labels": torch.tensor(pred_labels, dtype=torch.int64),
                    }
                )

            summary = evaluate_yolo_style(
                preds=final_preds,
                targets=all_targets,
                iou_thresholds=list(np.arange(0.5, 0.96, 0.05)),
                conf_thresholds=RUN_CONFIG["conf_thresholds"],
            )
            log_results_to_mlflow(summary, ID_TO_CLASS=ID_TO_CLASS)


if __name__ == "__main__":
    main()
