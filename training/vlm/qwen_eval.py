from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info

import os
import yaml
import json
import glob
from pathlib import Path
import mlflow
import torch
from PIL import Image

# from torchmetrics.detection import MeanAveragePrecision
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

from util.metrics import (
    evaluate_yolo_style,
    log_yolo_metrics_to_mlflow,
    draw_and_log_predictions_to_mlflow,
)

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


def run_qwen_inference(image: str, processor, model):
    # mostly from https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

    img = Image.open(image).convert("RGB")

    def compute_qwen_resize(orig_w, orig_h):
        # target: multiple of 28, but preserve aspect ratio
        scale = min(
            (orig_w // 32 * 32) / orig_w,
            (orig_h // 32 * 32) / orig_h,
        )

        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)

        # snap to multiple of 32
        # 32 for Qwen3-vl and 28 for qwen2.5-vl
        # from https://github.com/QwenLM/Qwen3-VL
        resized_w = (resized_w // 32) * 32
        resized_h = (resized_h // 32) * 32

        return resized_w, resized_h, scale

    def scale_1000_to_pixels(box, img_w, img_h):
        x1, y1, x2, y2 = box
        return [
            x1 / 1000 * img_w,
            y1 / 1000 * img_h,
            x2 / 1000 * img_w,
            y2 / 1000 * img_h,
        ]

    resized_w, resized_h, scale = compute_qwen_resize(img.width, img.height)

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_width": resized_w,
                    "resized_height": resized_h,
                },
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]

    # inputs = tokenizer(query, return_tensors="pt").to(DEVICE)
    # inputs = processor.apply_chat_template(
    #     message,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_dict=True,
    #     return_tensors="pt",
    # ).to(model.device)
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(message, image_patch_size=16)

    inputs = processor(
        text=text, images=images, videos=videos, do_resize=False, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parse_output_to_json(output_text)

    boxes = []
    labels = []
    scores = []

    for item in parsed:
        if not isinstance(item, list) or len(item) != 3:
            continue

        label, score, box = item
        label = normalize_label(label)

        if label not in CLASS_TO_ID:
            continue
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [float(x) for x in box]

            # rescale boxes to match original size
            scale_x = img.width / resized_w
            scale_y = img.height / resized_h

            x1o = x1 * scale_x
            y1o = y1 * scale_y
            x2o = x2 * scale_x
            y2o = y2 * scale_y

            box = [x1o, y1o, x2o, y2o]

            # qwen assume image is 0-1000
            box = scale_1000_to_pixels(box, img.width, img.height)

            # print("orig:", img.width, img.height)
            # print("resized:", resized_w, resized_h)
            # print("raw box:", x1, y1, x2, y2)
            # print("scaled box:", x1o, y1o, x2o, y2o)
            # print(box)

            score = float(score)
        except Exception:
            # print(e)
            continue

        boxes.append(box)
        labels.append(CLASS_TO_ID[label])
        scores.append(score)

    if len(boxes) == 0:
        return [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
        ]

    return [
        {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    ]


def main():
    """Main Function

    1. Load dataset
    2. Load model and processor
    3. Generate output per image with text prompt
    4. Parse output to JSON
    5. Calculate metrics
    """
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_paths = sorted(image_paths)
    RUN_CONFIG["valid_images"] = len(image_paths)

    print(json.dumps(RUN_CONFIG, indent=4))
    # tokenizer = AutoTokenizer.from_pretrained(
    #     RUN_CONFIG["model_id"], trust_remote_code=True
    # )

    # model = (
    #     AutoModelForCausalLM.from_pretrained(
    #         RUN_CONFIG["model_id"], trust_remote_code=True, bf16=True
    #     )
    #     .to(DEVICE)
    #     .eval()
    # )

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
            # img_path = Path(img_path)
            image = Image.open(img_path).convert("RGB")
            label_path = os.path.join(
                LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            )
            label = load_yolo_labels(label_path, image.width, image.height)

            pred = run_qwen_inference(img_path, processor, model)[0]
            target = prepare_targets(label)[0]

            all_preds.append(pred)
            all_targets.append(target)

        summary = evaluate_yolo_style(
            preds=all_preds,
            targets=all_targets,
            num_classes=len(CLASS_NAMES),
            iou_thresholds=np.arange(0.50, 0.96, 0.05),  # mAP50-95
        )

        log_yolo_metrics_to_mlflow(
            summary=summary,
            class_names=CLASS_NAMES,
            map_label="mAP50-95",
        )

        if RUN_CONFIG.get("store_images_and_pred", False):
            draw_and_log_predictions_to_mlflow(
                preds=all_preds,
                image_paths=image_paths,
                class_names=CLASS_NAMES,
                artifact_dir="prediction_images",
                score_threshold=RUN_CONFIG["conf_thresh"],
            )


if __name__ == "__main__":
    main()
