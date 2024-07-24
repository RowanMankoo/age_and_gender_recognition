MAX_LINE_SIZE := 120

export MODEL_FOLDER_PATH=tb_logs/production_model/version_2
# make sure to chnage this arg after running `make docker-push`
IMAGE_NAME ?= europe-west2-docker.pkg.dev/rowan-420019/apis/age-and-gender-apis@sha256:7dceef139b8bb49c53881f42981a28d1c615a8f77619facf4c6fbf6602ae11b0

make pre-lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) **/*.py
	poetry run flake8 --max-line-length $(MAX_LINE_SIZE) **/*.py

make lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) --check .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) --check-only **/*.py
	poetry run flake8 --max-line-length=$(MAX_LINE_SIZE) **/*.py

make tensorboard:
	poetry run tensorboard --logdir=tb_logs

make start-api-no-docker:
	uvicorn serving.api:app --host 0.0.0.0 --port 8080

make docker-build:
	docker build --build-arg MODEL_FOLDER_PATH=$(MODEL_FOLDER_PATH) -t age-and-gender-api -f serving/Dockerfile .

make docker-run:
	docker run -p 8080:8080 age-and-gender-api

make docker-tag:
	docker tag age-and-gender-api europe-west2-docker.pkg.dev/rowan-420019/apis/age-and-gender-apis

make docker-push:
	docker push europe-west2-docker.pkg.dev/rowan-420019/apis/age-and-gender-apis

deploy-cloud-run:
	gcloud run deploy --image=$(IMAGE_NAME) --memory 1Gi