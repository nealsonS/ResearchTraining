# Research Training
Just a repo for training scripts for my Research

# The Setup (Optimized for Cost Efficiency)
1. Use a VPS with GPU access with Runpod and upload data there to a persistent storage to make training easy
2. On local device (since it's only me that's training), host mlflow to 0.0.0.0
3. Use tailscale to connect both VPS and local device to version models