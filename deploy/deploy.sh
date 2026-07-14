#!/usr/bin/env bash
#
# deploy.sh — build the current checkout, deploy it, verify it BOOTS *and*
# SERVES, and auto-roll-back if verification fails.
#
#   Usage:  deploy/deploy.sh [--pull]
#     --pull   git pull the source checkout (fast-forward only) before building
#
# Encodes the rules learned the hard way in the July-2026 OCR outage:
#   • BUILD from the source checkout with `docker build` — NEVER
#     `docker compose build`, which tags the image with that dir's .env
#     IMAGE_TAG and would clobber your rollback image.
#   • DEPLOY from the runtime dir with an EXPLICIT IMAGE_TAG in .env — never
#     the bare `:latest` default (that is how a known-broken build went live).
#   • SNAPSHOT the running image before switching, so rollback is instant.
#   • VERIFY /healthz AND a live OCR request. A green healthcheck alone is NOT
#     proof the service works — that is exactly how the incident shipped.
#
set -euo pipefail

SRC_DIR="${SRC_DIR:-/var/www/kyc-test}"   # git checkout w/ Dockerfile — where we BUILD
RUN_DIR="${RUN_DIR:-$HOME/kyc-ocr}"       # compose dir — where we DEPLOY / RUN
IMAGE="recriviodev/kyc-ocr"
CONTAINER="kyc-ocr"
PORT="127.0.0.1:8000"
SAMPLE="$SRC_DIR/sample/pan-test.png"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

c_blue='\033[1;34m'; c_red='\033[1;31m'; c_grn='\033[1;32m'; c_off='\033[0m'
log()  { printf "${c_blue}[deploy]${c_off} %s\n" "$*"; }
ok()   { printf "${c_grn}[deploy]${c_off} %s\n" "$*"; }
die()  { printf "${c_red}[deploy:FATAL]${c_off} %s\n" "$*" >&2; exit 1; }

[ -d "$SRC_DIR" ] || die "source dir $SRC_DIR not found"
[ -d "$RUN_DIR" ] || die "runtime dir $RUN_DIR not found"
[ -f "$RUN_DIR/docker-compose.yml" ] || die "no docker-compose.yml in $RUN_DIR"
[ -f "$SAMPLE" ] || die "sample $SAMPLE missing (needed to verify OCR path)"

if [ "${1:-}" = "--pull" ]; then
  log "git pull --ff-only in $SRC_DIR"
  git -C "$SRC_DIR" pull --ff-only
fi

SHA="$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
DIRTY=""; git -C "$SRC_DIR" diff --quiet 2>/dev/null || DIRTY=" (UNCOMMITTED changes)"
STAMP="$(date +%Y-%m-%d-%H%M)"
NEW_TAG="live-$STAMP-$SHA"
log "building git $SHA$DIRTY  ->  $IMAGE:$NEW_TAG"

# 1) free disk — the image-export step failed on us once at 88% full
log "pruning build cache…"
docker builder prune -af >/dev/null 2>&1 || true
FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
[ "${FREE_GB:-0}" -ge "$MIN_FREE_GB" ] \
  || die "only ${FREE_GB}G free on / (need ${MIN_FREE_GB}G). Remove old images: docker images '$IMAGE'"

# 2) snapshot the CURRENTLY RUNNING image as a rollback point
CUR_IMG="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || true)"
if [ -n "$CUR_IMG" ]; then
  docker tag "$CUR_IMG" "$IMAGE:rollback-pre-$STAMP"
  echo "$CUR_IMG" > "$RUN_DIR/.last-image"
  log "snapshot: $CUR_IMG -> $IMAGE:rollback-pre-$STAMP"
else
  log "no running container — treating as first deploy (no snapshot)"
fi

# 3) build with an EXPLICIT tag (never `docker compose build`)
docker build -t "$IMAGE:$NEW_TAG" "$SRC_DIR" || die "docker build failed"

# 4) deploy from the runtime dir with an EXPLICIT IMAGE_TAG; recreate by name
printf 'IMAGE_TAG=%s\n' "$NEW_TAG" > "$RUN_DIR/.env"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
( cd "$RUN_DIR" && docker compose up -d ) || die "docker compose up failed"

# 5) VERIFY it boots (healthy) AND serves (real OCR → 200)
log "verifying (health + live OCR request)…"
for _ in $(seq 1 30); do
  H="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo none)"
  case "$H" in healthy|unhealthy) break;; esac
  sleep 5
done
H="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo none)"
CODE="$(curl -s -o /dev/null -w '%{http_code}' -F "file=@$SAMPLE" "http://$PORT/api/v1/ocr/pan" 2>/dev/null || echo 000)"

if [ "$H" = "healthy" ] && [ "$CODE" = "200" ]; then
  ok "LIVE — $IMAGE:$NEW_TAG  (health=$H, pan=$CODE, git=$SHA)"
  exit 0
fi

# 6) verification failed — auto roll back to the image we snapshotted
log "VERIFY FAILED (health=$H, pan=$CODE) — rolling back"
docker logs --tail=30 "$CONTAINER" 2>&1 | sed 's/^/    /' || true
if [ -n "$CUR_IMG" ]; then
  printf 'IMAGE_TAG=%s\n' "${CUR_IMG##*:}" > "$RUN_DIR/.env"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  ( cd "$RUN_DIR" && docker compose up -d ) || true
  die "rolled back to $CUR_IMG. Broken image $IMAGE:$NEW_TAG kept for debugging."
fi
die "no previous image to roll back to — fix the build and redeploy."
