import glob
import logging
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from src.modelling import MultiTaskNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

model_folder_path = Path(os.environ["MODEL_FOLDER_PATH"])
ckpt_files = glob.glob(str(model_folder_path / "checkpoints/*.ckpt"))
hparams_file = model_folder_path / Path("hparams.yaml")


def crop_face(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(image_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, return the original image
    if len(faces) == 0:
        return image

    x, y, w, h = faces[0]
    face = image_cv[y : y + h, x : x + w]

    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    return face_pil


if ckpt_files:
    model = MultiTaskNet.load_from_checkpoint(
        ckpt_files[0],
        production_model=True,
    )
    model.to("cpu")
    model.eval()
else:
    raise FileNotFoundError("No checkpoint files found in the model folder")

with open(hparams_file, "r") as f:
    data = yaml.safe_load(f)

DATA_MEANS, DATA_STD = torch.tensor(data["data_means"]), torch.tensor(data["data_std"])

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_MEANS, std=DATA_STD),
    ]
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    logger.info("Received a prediction request")

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image = crop_face(image)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to("cpu")

    with torch.no_grad():
        gender_out, age_out = model(image)

    gender = "Male" if torch.argmax(gender_out) == 0 else "Female"
    age = torch.round(age_out).item()

    logger.info(f"Returning prediction: {gender}, {age}")

    return {"gender": gender, "age": age}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
