# Training files
- To be run from the VM
- SSH into the VM

# Tailscale setup
- Get Auth Key from `https://login.tailscale.com/admin/settings/keys`
  - Set it as `Ephemeral`, so it detaches the device when finishing
- Copy `.env.example` to `.env` and fill in the `TAILSCALE_AUTH_KEY` variable with your Auth Key
- Run `./tailscale_setup.sh`
  - Start using userspace networking mode from https://tailscale.com/docs/concepts/userspace-networking#start-tailscale-in-userspace-networking-mode

# Training scripts
- Then run the script