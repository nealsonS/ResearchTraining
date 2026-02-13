# Tracking Server

Uses a local MLFLow that runs in `0.0.0.0:5000`.

# To Setup
Just `docker compose up -d`
- Give it proper permissions using `chmod`
- Volumes will be persisted 
- Copy `.env.example` to create your own `.env`
- Please add `TAILSCALE_DNS` environment variable to the .env file!!! to allow connections to the mlflow server from tailscale!

# Additions
> The docker-compose is taken from `https://github.com/mlflow/mlflow.git` but editted.

- Added `--allowed-hosts "localhost:5000,${TAILSCALE_DNS}:5000"` to allow for tailscale connections.