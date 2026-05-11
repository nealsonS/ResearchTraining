"""Draw predictions (and optionally ground-truth boxes) on images from a predictions CSV.

CSV schema (produced by log_predictions_to_mlflow):
    image, type, x1, y1, x2, y2, score, label
    type is either "pred" or "gt"

Usage:
    python draw_predictions.py predictions.csv --output-dir drawn/
    python draw_predictions.py predictions.csv --output-dir drawn/ --conf 0.5 --no-gt
    python draw_predictions.py predictions.csv --output-dir drawn/ --images 3
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PRED_COLOR = "red"
GT_COLOR = "lime"
FONT_SIZE = 14


def _get_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(image_path: str, rows: pd.DataFrame, font) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    for _, row in rows.iterrows():
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        color = PRED_COLOR if row["type"] == "pred" else GT_COLOR
        width = 2 if row["type"] == "pred" else 1

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = str(row["label"])
        if row["type"] == "pred" and pd.notna(row.get("score")):
            label = f"{label} {row['score']:.2f}"

        text_y = max(0, y1 - FONT_SIZE - 2)
        draw.text((x1, text_y), label, fill=color, font=font)

    return im


def main():
    parser = argparse.ArgumentParser(description="Draw predictions on images")
    parser.add_argument("csv", help="Path to predictions CSV")
    parser.add_argument("--output-dir", "-o", default="drawn_predictions", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.0, help="Min confidence threshold for pred boxes")
    parser.add_argument("--no-gt", action="store_true", help="Skip ground-truth boxes")
    parser.add_argument("--no-pred", action="store_true", help="Skip prediction boxes")
    parser.add_argument("--images", type=int, default=None, help="Limit number of images to draw")
    parser.add_argument("--image-root", default=None, help="Override image root path prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.conf > 0:
        df = df[~((df["type"] == "pred") & (df["score"] < args.conf))]
    if args.no_gt:
        df = df[df["type"] != "gt"]
    if args.no_pred:
        df = df[df["type"] != "pred"]

    image_paths = df["image"].unique()
    if args.images is not None:
        image_paths = image_paths[: args.images]

    font = _get_font()
    drawn, skipped = 0, 0

    for img_path in image_paths:
        resolved = Path(img_path) if args.image_root is None else Path(args.image_root) / Path(img_path).name
        if not resolved.exists():
            print(f"[skip] {resolved}")
            skipped += 1
            continue

        rows = df[df["image"] == img_path]
        im = draw_boxes(str(resolved), rows, font)
        out_path = output_dir / (Path(img_path).stem + ".png")
        im.save(out_path)
        drawn += 1

    print(f"Saved {drawn} image(s) to {output_dir}/  ({skipped} skipped)")


if __name__ == "__main__":
    main()
