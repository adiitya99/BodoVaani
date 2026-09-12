from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import os

from app.bodo_vaani import BodoVaani


app = FastAPI(
    title="BodoVaani API",
    description="Bidirectional Bodo ↔ English Translation",
    version="1.0.0",
)


bodo_vaani = BodoVaani()


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    direction: str = Form("bodo_to_english"),
):

    os.makedirs("audio/uploads", exist_ok=True)

    temp_path = "audio/uploads/uploaded_audio.wav"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if direction == "bodo_to_english":

        result = bodo_vaani.process_bodo_to_english(
            temp_path
        )

        return result

    return {
        "error": "Unsupported direction"
    }


@app.post("/translate")
def translate(
    text: str = Form(...),
    direction: str = Form("english_to_bodo"),
):

    if direction == "english_to_bodo":

        result = bodo_vaani.process_english_to_bodo(
            text
        )

        return result

    return {
        "error": "Unsupported direction"
    }
