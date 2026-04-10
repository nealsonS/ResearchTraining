#!/bin/bash
source .env
curl -fsSL https://tailscale.com/install.sh | sh

tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 &
tailscale up --auth-key=${TAILSCALE_AUTH_KEY}

export ALL_PROXY=socks5h://127.0.0.1:1055