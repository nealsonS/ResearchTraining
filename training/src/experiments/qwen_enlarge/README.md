# Qwen Experiment

## Experiment Details
Used VLM Qwen to perform object detection on 

## Caveats
- Qwen expects images to be of size 1000 (both x and y axis)
  - so have to rescale them
- Qwen-3b also reshapes images to size 32
  - so adjust for it when rescaling bbox at the end