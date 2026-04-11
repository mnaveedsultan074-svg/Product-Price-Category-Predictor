#!/usr/bin/env bash
# Build and push multi-arch images (linux/amd64 + linux/arm64) to Docker Hub.
#
# Usage:
#   export DOCKERHUB_USER=isammalik   # optional if detectable from Docker config
#   ./bin/push-dockerhub.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDER_NAME="${DOCKER_BUILDX_BUILDER:-price-predictor-multiarch}"
ML_IMAGE="product-price-predictor-ml-api"
RAILS_IMAGE="product-price-predictor-rails-app"

detect_dockerhub_user() {
  python3 <<'PY'
import base64
import json
import os
import shutil
import subprocess
import sys

def user_from_auth_block(block):
    auth = (block or {}).get("auth")
    if not auth:
        return None
    try:
        raw = base64.b64decode(auth).decode("utf-8", errors="replace")
        user = raw.split(":", 1)[0].strip()
        if user and user.lower() != "oauth2accesstoken":
            return user
    except Exception:
        return None
    return None

def credential_get(binary: str, registry_url: str):
    r = subprocess.run(
        [binary, "get"],
        input=registry_url + "\n",
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    data = json.loads(r.stdout)
    return data.get("Username") or data.get("username")

def main():
    path = os.path.expanduser("~/.docker/config.json")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return None

    auths = cfg.get("auths") or {}
    auth_keys = [
        "https://index.docker.io/v1/",
        "https://index.docker.io/v1",
        "registry-1.docker.io",
        "https://registry-1.docker.io",
        "docker.io",
        "https://docker.io/v1/",
    ]
    for k in auth_keys:
        block = auths.get(k) or auths.get(k.rstrip("/"))
        u = user_from_auth_block(block)
        if u:
            return u

    cred_helpers = cfg.get("credHelpers") or {}
    helper = (
        cred_helpers.get("docker.io")
        or cred_helpers.get("https://index.docker.io/v1/")
        or cred_helpers.get("registry-1.docker.io")
    )
    if helper:
        binary = f"docker-credential-{helper}"
        if shutil.which(binary):
            for url in (
                "https://index.docker.io/v1/",
                "registry-1.docker.io",
            ):
                try:
                    u = credential_get(binary, url)
                    if u:
                        return u
                except Exception:
                    continue

    store = cfg.get("credsStore")
    if store:
        binary = f"docker-credential-{store}"
        if shutil.which(binary):
            for url in (
                "https://index.docker.io/v1/",
                "https://index.docker.io/v1",
                "https://registry-1.docker.io",
                "registry.hub.docker.com",
            ):
                try:
                    u = credential_get(binary, url)
                    if u:
                        return u
                except Exception:
                    continue

    return None

if __name__ == "__main__":
    u = main()
    if u:
        print(u)
    else:
        sys.exit(1)
PY
}

if [[ -z "${DOCKERHUB_USER:-}" ]]; then
  if u="$(detect_dockerhub_user)"; then
    export DOCKERHUB_USER="$u"
    echo "Detected DOCKERHUB_USER=$DOCKERHUB_USER (from ~/.docker/config.json helpers/auths)."
  else
    echo "Set DOCKERHUB_USER to your Docker Hub namespace, e.g.:" >&2
    echo "  export DOCKERHUB_USER=isammalik" >&2
    exit 1
  fi
else
  echo "Using DOCKERHUB_USER=$DOCKERHUB_USER (from environment)."
fi

echo ""
echo "Ensuring buildx builder: $BUILDER_NAME"
if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx use "$BUILDER_NAME"
else
  docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
fi
docker buildx inspect --bootstrap >/dev/null

echo ""
echo "Building & pushing ${DOCKERHUB_USER}/${ML_IMAGE}:latest ..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --provenance=false \
  --push \
  -t "${DOCKERHUB_USER}/${ML_IMAGE}:latest" \
  -f "${REPO_ROOT}/ml/Dockerfile" \
  "${REPO_ROOT}/ml"

echo ""
echo "Building & pushing ${DOCKERHUB_USER}/${RAILS_IMAGE}:latest ..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --provenance=false \
  --push \
  -t "${DOCKERHUB_USER}/${RAILS_IMAGE}:latest" \
  -f "${REPO_ROOT}/price_predictor/Dockerfile" \
  "${REPO_ROOT}/price_predictor"

echo ""
echo "Done. Set each repository to Public on Docker Hub so anonymous pulls work:"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/${ML_IMAGE}/settings"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/${RAILS_IMAGE}/settings"
echo ""
echo "Pull and run:"
echo "  export DOCKERHUB_USER=${DOCKERHUB_USER}"
echo "  docker compose -f docker-compose.hub.yml pull && docker compose -f docker-compose.hub.yml up"
