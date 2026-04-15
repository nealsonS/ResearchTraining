# Visual Language Models

- Directory for VLM and evaluation.
- All scripts depend on `eval_config.yaml`. Please edit this to change run configurations
- Additionally, all results will be logged using `MLFlow` to the tracking server highlighted in `../.env` file
- Current scripts:
  - `dino.py`: Grounding Dino Inference
  - `qwen.py`: For VLMs that is chat-based or doesn't have a fixed output
  - `yolo_train.py`: Training Yolo model on dataset
    - WIP
  - `yolo_qwen.py`: Training YOLO model on logo (any logo) detection, crop the bounding box's image, then ask Qwen to classify the logo
