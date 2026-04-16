import torch
from ResearchTraining.util.io import normalize_label


def generate_dino_labels(classes: list[str]) -> str:
    """For each label, generate the text prompt for each label"""
    text_prompt = ""

    for cls in classes:
        text_prompt += f"{cls}. "

    return text_prompt.strip()


def run_grounding_dino(
    image, processor, model, text_prompt: str, class_to_id: dict[str, int]
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        # threshold=CONF_THRESH,
        target_sizes=[(image.height, image.width)],
    )[0]

    boxes = []
    scores = []
    labels = []

    for box, score, label in zip(
        results["boxes"], results["scores"], results["labels"]
    ):
        label = normalize_label(label)

        if label not in class_to_id:
            continue

        boxes.append(box.detach().cpu())
        scores.append(score.detach().cpu())
        labels.append(class_to_id[label])

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
            "boxes": torch.stack(boxes).to(torch.float32),
            "scores": torch.stack(scores).to(torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    ]
