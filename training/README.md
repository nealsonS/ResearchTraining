# Training files
- To be run from the VM
- SSH into the VM

# Tailscale setup
- Get Auth Key from `https://login.tailscale.com/admin/settings/keys`
  - Set it as `Ephemeral`, so it detaches the device when finishing
- Copy `.env.example` to `.env` and fill in the `TAILSCALE_AUTH_KEY` variable with your Auth Key

- If running on a cluster/not local computer:
  - Run `./tailscale_setup.sh`
    - Start using userspace networking mode from https://tailscale.com/docs/concepts/userspace-networking#start-tailscale-in-userspace-networking-mode

# Training scripts
- Then run the script

## Runpod Caveats
- Check CUDA version using `nvidia-smi` and install torch version that matches cuda version
  - add to the requirements.txt
  - for example: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- Since Runpod's tailscale is in userspace networking mode, need to route to proxy server first
  - Run `export ALL_PROXY=socks5h://127.0.0.1:1055`