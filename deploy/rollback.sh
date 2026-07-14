#!/usr/bin/env bash
#
# rollback.sh — instantly point the runtime at a known-good image and recreate
# the container. NEVER rebuilds, so it works even when the source/build is broken.
#
#   rollback.sh                       roll back to the image running before the last deploy
#   rollback.sh <tag>                 roll back to a specific tag (e.g. rollback-pre-2026-07-10)
#   rollback.sh --list                list local images you can roll back to
#
# Pairs with deploy.sh, which writes $RUN_DIR/.last-image (the pre-deploy image)
# and snapshots each running image as rollback-pre-<stamp> before switching.
#
set -euo pipefail

RUN_DIR="${RUN_DIR:-$HOME/kyc-ocr}"
IMAGE="recriviodev/kyc-ocr"
CONTAINER="kyc-ocr"
PORT="127.0.0.1:8000"
SAMPLE="${SAMPLE:-/var/www/kyc-test/sample/pan-test.png}"

c_blue='\033[1;34m'; c_red='\033[1;31m'; c_off='\033[0m'
log() { printf "${c_blue}[rollback]${c_off} %s\n" "$*"; }
die() { printf "${c_red}[rollback:FATAL]${c_off} %s\n" "$*" >&2; exit 1; }

if [ "${1:-}" = "--list" ]; then
  docker images "$IMAGE" --format 'table {{.Tag}}\t{{.CreatedSince}}\t{{.Size}}'
  exit 0
fi

if [ -n "${1:-}" ]; then
  TAG="$1"
elif [ -f "$RUN_DIR/.last-image" ]; then
  TAG="$(sed 's/.*://' "$RUN_DIR/.last-image")"
  log "no tag given — using the pre-deploy image tag: $TAG"
else
  die "no tag given and no $RUN_DIR/.last-image present; run: $0 --list"
fi

docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1 \
  || die "image $IMAGE:$TAG not found locally. Choose one from: $0 --list"

log "rolling back to $IMAGE:$TAG"
printf 'IMAGE_TAG=%s\n' "$TAG" > "$RUN_DIR/.env"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
( cd "$RUN_DIR" && docker compose up -d )

for _ in $(seq 1 30); do
  H="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo none)"
  case "$H" in healthy|unhealthy) break;; esac
  sleep 5
done
H="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo none)"
CODE="$(curl -s -o /dev/null -w '%{http_code}' -F "file=@$SAMPLE" "http://$PORT/api/v1/ocr/pan" 2>/dev/null || echo 000)"
log "health=$H  pan=$CODE"
if [ "$H" = "healthy" ] && [ "$CODE" = "200" ]; then
  log "rollback OK — $IMAGE:$TAG is live and serving"
else
  die "rolled to $TAG but it did NOT verify (health=$H pan=$CODE). Try another: $0 --list"
fi
