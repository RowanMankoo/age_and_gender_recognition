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

ARG MODEL_FOLDER_PATH
ENV MODEL_FOLDER_PATH=${MODEL_FOLDER_PATH}

WORKDIR /app
COPY src/ /app/src
COPY serving/ /app/serving
COPY ${MODEL_FOLDER_PATH} /app/${MODEL_FOLDER_PATH}

EXPOSE 80

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080"]