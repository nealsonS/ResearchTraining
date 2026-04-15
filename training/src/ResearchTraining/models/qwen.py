import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from ResearchTraining.util.io import parse_output_to_json, normalize_label


def run_qwen_inference(
    image: str, processor, model, text_prompt: str, class_to_id: dict[str, int]
):
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
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

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

        if label not in class_to_id:
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

            score = float(score)
        except Exception:
            continue

        boxes.append(box)
        labels.append(class_to_id[label])
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
