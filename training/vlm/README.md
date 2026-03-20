# Visual Language Models

- Directory for VLM and evaluation.
- All scripts depend on `eval_config.yaml`. Please edit this to change run configurations
- Additionally, all results will be logged using `MLFlow` to the tracking server highlighted in `../.env` file
- Current scripts:
  - `dino_eval.py`: Grounding Dino Inference
  - `vlm_eval.py`: For VLMs that is chat-based or doesn't have a fixed output
