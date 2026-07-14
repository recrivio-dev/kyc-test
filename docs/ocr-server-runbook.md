# KYC OCR Server — Runbook & Reference

Everything needed to understand, operate, deploy, and debug the in-house KYC OCR
service. Hand this to a new developer as their starting point.

> **Repo:** `github.com/recrivio-dev/kyc-test` · **Image:** `recriviodev/kyc-ocr` ·
> **Public URL:** `https://ocr.recriauth.com`

---

## 1. What it is

A standalone Python microservice that OCRs and redacts Indian KYC documents
(Aadhaar, PAN, Passport, Voter ID, Driving Licence). It **locates** text, **reads**
only the crops, **classifies** the document, extracts structured fields into a
stable JSON contract, and can return a **redacted** image.

- **Stack:** Python 3.11 · FastAPI served by **gunicorn** (2 uvicorn workers) ·
  OpenCV · RapidOCR (PP-OCR via ONNX Runtime) · optional Surya fallback (gated
  **off**) · Docker. No GPU, no PaddlePaddle.
- **Consumed by:** the main `recriauth` backend as the `IN_HOUSE_OCR` vendor.
  ⚠️ **recriauth silently falls back to the paid SurePass vendor on ANY OCR
  failure** — so if this service breaks, verifications keep working but vendor
  cost rises and **nothing pages**. Treat OCR health as your responsibility to
  watch; do not rely on user reports.
- **Deep dives:** pipeline internals → [`README.md`](../README.md); API examples →
  [`postman.md`](../postman.md); the July-2026 outage → [`docs/postmortem-2026-07-ocr-500.md`](postmortem-2026-07-ocr-500.md).

---

## 2. Infrastructure / topology

| Piece | Detail |
|---|---|
| **Host** | AWS EC2, `13.235.84.85` (a.k.a. `ec2-13-235-84-85.ap-south-1.compute.amazonaws.com`), Ubuntu, x86_64, ~4 GB RAM, **28 GB root disk** |
| **SSH** | `ssh -i recrivio-raj.pem ubuntu@13.235.84.85` (key + passwords live in the prod KT doc / secrets store — not in git) |
| **DNS** | `ocr.recriauth.com` → this box |
| **nginx (host)** | Terminates TLS (Let's Encrypt), proxies `:443` → `127.0.0.1:8000`. `client_max_body_size 25m`, 180s read/send timeouts. Live config: `/etc/nginx/sites-enabled/ocr.recriauth.com` (repo copy: [`deploy/nginx.conf`](../deploy/nginx.conf)) |
| **Container** | name `kyc-ocr`, image `recriviodev/kyc-ocr:<tag>`, binds **`127.0.0.1:8000`** only, gunicorn 2 workers, `restart: unless-stopped`, **mem limit 3 GB** |
| **Registry** | Docker Hub `recriviodev/kyc-ocr`. PATs: `ec2-kyc-pull` (pull), `mac-rajkunwar-push` (push) — in the KT doc |

### Two directories on the box — know the difference

| Dir | Role |
|---|---|
| **`/var/www/kyc-test`** | Git checkout **with the Dockerfile** — where you **BUILD**. Its `docker-compose.yml` has a `build:` section (a local, uncommitted edit). |
| **`~/kyc-ocr`** | Runtime compose dir — where the container **RUNS**. Image-only compose + a **`.env` that pins `IMAGE_TAG`**. |
| `~/kyc-deploy` | Abandoned/empty — ignore it. |

Both compose files declare `container_name: kyc-ocr`, so they're separate Docker
Compose *projects* fighting over one name. **Always build in `/var/www/kyc-test`
and run from `~/kyc-ocr`.**

---

## 3. The API

Base URL through nginx: `https://ocr.recriauth.com/api/v1/ocr`
(recriauth's `IN_HOUSE_OCR_BASE_URL`). All OCR endpoints are `POST` /
`multipart/form-data` with a single `file` field.

| Method | Path | Notes |
|---|---|---|
| GET | `/healthz` | 200 = models loaded **and** startup pipeline self-test passed; **503** = broken. Body includes `selftest_error`. |
| POST | `/api/v1/ocr/pan` `/aadhaar` `/passport` `/voter-id` `/driving-license` | Per-doc-type; returns `{ data: { ocr_fields: [...] } }` (DL is flat, not `ocr_fields`) |
| POST | `/api/v1/ocr` | Generic; add form field `doc_type=PAN\|AADHAAR\|PASSPORT\|VOTER_ID\|DRIVING_LICENSE` |
| POST | `/api/v1/ocr/mask-identity` | Redaction; form `document_type` + `file`. Returns `masked_image` + `unmasked_image` + `document_detected` + `masked_regions` |

Response envelope (all endpoints): `{ "data": {...}, "status_code", "message_code", "message", "success" }`.
Field/response shapes are documented per-doc-type in [`README.md`](../README.md).

Quick smoke test from the box:
```bash
curl -s -F "file=@/var/www/kyc-test/sample/pan-test.png" \
     http://127.0.0.1:8000/api/v1/ocr/pan | jq
curl -s http://127.0.0.1:8000/healthz | jq        # {"ok":true,"selftest_ok":true,...}
```

---

## 4. Deployment model

**Build on the server, run from the runtime dir.** Code lives in git; you pull it
onto `/var/www/kyc-test`, build an image tagged by date/SHA, and point
`~/kyc-ocr` at that tag via `IMAGE_TAG` in its `.env`. The compose line is:

```yaml
image: recriviodev/kyc-ocr:${IMAGE_TAG:-latest}
```

So the running version is **entirely controlled by `~/kyc-ocr/.env`**. Cross-arch
note: the server is amd64; if you ever build on an Apple-Silicon Mac instead, you
**must** `docker buildx --platform linux/amd64` (a plain build produces an
arm64 image that won't run here).

### Deploy — the easy way (use the scripts)

```bash
cd /var/www/kyc-test && git pull        # or let the script do it:
deploy/deploy.sh --pull                 # pull → build → snapshot → deploy → verify → auto-rollback on fail
```
`deploy/deploy.sh` ([source](../deploy/deploy.sh)):
1. prunes build cache; refuses to build under 6 GB free,
2. snapshots the **currently running** image as `rollback-pre-<stamp>`,
3. `docker build -t recriviodev/kyc-ocr:live-<stamp>-<sha>` from the checkout,
4. writes `IMAGE_TAG` to `~/kyc-ocr/.env`, recreates the container,
5. verifies it's `healthy` **and** a real `/api/v1/ocr/pan` returns 200,
6. **auto-rolls-back** and keeps the broken image for debugging if verify fails.

### Deploy — manual (if you're not using the script)

```bash
cd /var/www/kyc-test && git pull
docker build -t recriviodev/kyc-ocr:live-$(date +%F) .   # NOT `docker compose build` — see gotchas
cd ~/kyc-ocr
printf 'IMAGE_TAG=live-%s\n' "$(date +%F)" > .env
docker rm -f kyc-ocr && docker compose up -d
# verify:
docker inspect --format 'health={{.State.Health.Status}} restarts={{.RestartCount}}' kyc-ocr
curl -s -o /dev/null -w 'pan=%{http_code}\n' -F "file=@/var/www/kyc-test/sample/pan-test.png" http://127.0.0.1:8000/api/v1/ocr/pan
```
Success = `health=healthy`, `restarts=0`, `pan=200` with a real JSON body.

---

## 5. Rollback

```bash
deploy/rollback.sh                       # back to the image running before the last deploy
deploy/rollback.sh rollback-pre-2026-07-10   # to a specific tag
deploy/rollback.sh --list                # list available images
```
Manual equivalent:
```bash
printf 'IMAGE_TAG=rollback-pre-2026-07-10\n' > ~/kyc-ocr/.env && docker rm -f kyc-ocr && docker compose up -d
```
Rollback **never rebuilds** — it just flips `IMAGE_TAG` and recreates, so it works
even when the source/build is broken. Keep at least one known-good
`rollback-pre-*` image around at all times.

---

## 6. Gotchas / footguns (learned the hard way — read before you deploy)

1. **`~/kyc-ocr/.env` is the source of truth for the running version.** A bare
   `docker compose up -d` with no `IMAGE_TAG` resolves to `:latest` — which may be
   a broken build. Always pin the tag.
2. **Never `docker compose build` in `/var/www/kyc-test`.** Its `.env` pins a
   rollback tag, so `compose build` would tag the new image as that rollback tag
   and **clobber your rollback image**. Use `docker build -t <explicit tag> .`
   (the scripts do this).
3. **A "healthy" container is not proof it works — but it's much better now.**
   `/healthz` runs a real pipeline self-test at startup and returns 503 if the
   pipeline raises. Still, after any deploy, also `curl` a real OCR request.
4. **Pin every dependency, including transitive + base image:**
   - `rapidocr-onnxruntime` pulls the full `opencv-python` **unpinned** — pin it
     (`requirements.txt`), or its `HoughLinesP` shape drifts between builds and
     500s the OCR path.
   - Base image `python:3.11-slim` is a **moving tag**; it drifted to Debian 13
     (trixie) which lacks `libGL.so.1` → `import cv2` crashes at boot. Pinned to
     `python:3.11-slim-bookworm`.
   - `surya-ocr==0.17.0` **hard-pins** `opencv-python-headless==4.11.0.86`, so you
     **cannot** bump the headless OpenCV without changing surya.
5. **Disk is tight (28 GB root).** Each image ~3 GB; build needs ~6 GB free for
   the export step or it fails mid-build. A weekly `docker builder prune -af`
   cron is installed; free images with `docker rmi` if you run low.
6. **recriauth's SurePass fallback is silent.** An OCR outage produces zero
   alerts — check `docker logs` and vendor cost, don't wait to be told.

---

## 7. Common operations & troubleshooting

```bash
# What's live / running?
docker compose ps                                              # (from ~/kyc-ocr)
docker inspect --format '{{.Config.Image}}' kyc-ocr
cat ~/kyc-ocr/.env
docker logs -f --tail=80 kyc-ocr

# Health
curl -s http://127.0.0.1:8000/healthz | jq                     # selftest_ok / selftest_error
docker inspect --format '{{.State.Health.Status}}' kyc-ocr
```

| Symptom | Likely cause / fix |
|---|---|
| Container `restarting` / `unhealthy`, high `restarts` | Boot crash. `docker logs` → `ImportError` usually means a missing system lib or dep skew (e.g. `libGL.so.1` from base-image drift → ensure bookworm base). |
| `healthz` 503 with `selftest_error` | The pipeline raises on a real doc. The error text is the traceback tail — fix the code path, redeploy. |
| OCR returns 500 but container "healthy" | Shouldn't happen now (deep healthcheck). If it does, `docker logs` for the traceback; roll back. |
| `no space left on device` mid-build | `docker builder prune -af`; `docker rmi` old `live-*`/superseded images (keep a `rollback-pre-*`). |
| nginx 413 on upload | Body over 25 MB — raise `client_max_body_size` in the nginx site. |
| Requests time out ~120s | gunicorn `GUNICORN_TIMEOUT` / slow cold start; nginx `proxy_read_timeout` is 180s and must stay > gunicorn timeout. |

---

## 8. Config knobs (compose `environment` in `~/kyc-ocr/docker-compose.yml`)

| Var | Default | Notes |
|---|---|---|
| `KYC_ENABLE_SURYA` | `false` | Surya fallback OCR. Off — deps are brittle and it adds ~1.5–2 GB per worker. If you enable it, **drop `WEB_CONCURRENCY` to 1** on this 4 GB box. |
| `WEB_CONCURRENCY` | `2` | gunicorn workers (tuned for 2 vCPU). |
| `GUNICORN_TIMEOUT` | `120` | Covers cold-start model load + slow PDFs. |
| mem limit | `3g` | Container hard cap so the OS + nginx keep headroom. |

Model caches (`hf_cache`, `torch_cache`) are Docker volumes so restarts don't
re-download. `sample-docs` is a bind mount for masked-image output.

---

## 9. Local development

```bash
cd kyc-test
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000        # FastAPI, /docs for interactive testing
# or the whole container:
docker build -t recriviodev/kyc-ocr:dev . && docker run --rm -p 8000:8000 recriviodev/kyc-ocr:dev
```
⚠️ Dev machines often have the **full** `opencv-python` shadowing the headless
pin, so some bugs (like the `HoughLinesP` shape crash) only appear in the
container. Test against the built image before shipping, not just `uvicorn`.

---

## 10. Access & secrets (where they live — never commit them)

- **SSH key** `recrivio-raj.pem`, server passwords → prod KT doc / secrets store.
- **Docker Hub PATs** (`ec2-kyc-pull`, `mac-rajkunwar-push`) → KT doc. Log in on
  the box with `docker login -u recriviodev` before pull/push.
- **Server env** (`~/kyc-ocr` has no app secrets; the OCR service itself needs
  none — it's stateless). recriauth's `IN_HOUSE_OCR_BASE_URL` points here.
- Nothing in this repo should contain credentials.
