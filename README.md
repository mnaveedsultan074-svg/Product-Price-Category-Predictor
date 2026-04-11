# Product price category predictor

Rails web app (`price_predictor`) plus a Python ML API (`ml`) that predicts product price tiers. Docker Compose wires both services together.

## Prerequisites

- Docker with Compose v2
- For local training data download: Kaggle API credentials (see `.env.example`)

## Clone and run locally (build from this repo)

Tracked artefacts include `ml/models/` (metadata and paths for trained weights), `ml/data/processed/`, and application code. Large raw CSVs under `ml/data/raw/` stay ignored—download them via the pipeline or your own data.

```bash
cp .env.example .env
# Edit .env: set RAILS_MASTER_KEY, and optionally KAGGLE_USERNAME / KAGGLE_KEY for first-run training.

docker compose up --build
```

- ML API: [http://localhost:5001](http://localhost:5001) (health: `/health`)
- Web: [http://localhost:3000](http://localhost:3000)

On first boot the ML container runs download → preprocess → train → serve; later restarts reuse the `ml_models` and `ml_data` volumes.

### Database and storage volume

Production uses SQLite at `storage/production.sqlite3`. Compose mounts a named volume on `rails/storage`, so any database created during the image build is hidden at runtime. The container entrypoint runs `rails db:prepare` before `rails server` so migrations run on first boot.

## Run pre-built images from Docker Hub

Use the same ports, volumes, and health checks as local compose, but pull images instead of building:

```bash
export DOCKERHUB_USER=isammalik   # your Docker Hub username (example: maintainer account)

docker compose -f docker-compose.hub.yml pull
docker compose -f docker-compose.hub.yml up
```

Images are referenced as `${DOCKERHUB_USER}/product-price-predictor-ml-api:latest` and `${DOCKERHUB_USER}/product-price-predictor-rails-app:latest`—there is no hard-coded Hub namespace in the compose file.

For anonymous `docker pull` to work, set **Repository visibility → Public** for both repos on Docker Hub (after the first push creates them):

- `https://hub.docker.com/repository/docker/<username>/product-price-predictor-ml-api/settings`
- `https://hub.docker.com/repository/docker/<username>/product-price-predictor-rails-app/settings`

## Push multi-arch images to Docker Hub

From the repo root (requires `docker login` and [buildx](https://docs.docker.com/build/buildx/)):

```bash
export DOCKERHUB_USER=isammalik
./bin/push-dockerhub.sh
```

If `DOCKERHUB_USER` is unset, the script tries to infer your Hub username from `~/.docker/config.json` (`credHelpers`, `credsStore`, then `auths` base64). It builds **linux/amd64** and **linux/arm64** with `--provenance=false` and pushes:

- `<user>/product-price-predictor-ml-api:latest`
- `<user>/product-price-predictor-rails-app:latest`

## Optional: DistilBERT utilities

The default Docker/Compose path uses the sklearn / XGBoost / LightGBM stack. The standalone script `ml/distilbert_classifier.py` needs extra dependencies; install locally with `pip install torch transformers` if you use it.

## Layout

| Path | Role |
|------|------|
| `ml/` | Training pipeline, Flask `predict_service.py`, models and processed data |
| `price_predictor/` | Rails 7 app |
| `docker-compose.yml` | Local development: `build:` contexts |
| `docker-compose.hub.yml` | Production-style pull: `image:` only |
