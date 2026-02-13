# Tracking Server

Uses a local MLFLow that runs in `0.0.0.0:5000`.

# To Setup
Just `docker compose up -d`
- Give it proper permissions using `chmod`
- Volumes will be persisted 
- Copy `.env.example` to create your own `.env`

# Additions
> The docker-compose is taken from `https://github.com/mlflow/mlflow.git` but editted.

- Added `--allowed-hosts "mlflow.company.com,localhost:*"` to allow for tailscale connections.