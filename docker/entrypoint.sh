#!/bin/bash
# Container entrypoint: bring up sshd (idempotent), then exec the CMD.
#
# - Installs $PUBLIC_KEY to /root/.ssh/authorized_keys when set (RunPod convention).
# - Starts sshd in the background only if it isn't already running.
# - Skips silently when no key material is available, so local builds without
#   $PUBLIC_KEY still come up — sshd just won't accept connections.

set -e

if [[ -n "${PUBLIC_KEY:-}" && ! -s /root/.ssh/authorized_keys ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

mkdir -p /run/sshd
ssh-keygen -A >/dev/null 2>&1 || true

if ! pgrep -x sshd >/dev/null 2>&1; then
    /usr/sbin/sshd
fi

exec "$@"