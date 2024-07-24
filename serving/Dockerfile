FROM python:3.10-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY ./serving/pyproject.toml ./serving/poetry.lock* /app/serving/

WORKDIR /app/serving
RUN poetry install --no-interaction --no-ansi

ARG PATH_TO_MODEL_CHECKPOINT=tb_logs/production_model/version_2

WORKDIR /app
COPY src/ /app/src
COPY serving/ /app/serving
COPY ${PATH_TO_MODEL_CHECKPOINT} /app/${PATH_TO_MODEL_CHECKPOINT}

EXPOSE 80

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080"]