import logging
from typing import Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from align import init_ffmepg, init_models, create
from config import INPUT_DIR, OUTPUT_DIR
from utils import delete_file, file_ext, generate_unique_filename

app = FastAPI(root_path="/api")
app1 = FastAPI()
app1.mount(
    "/output",
    StaticFiles(directory="output"),
    name="output",
)

app.mount("/v1", app1)


class CreateVideoRequest(BaseModel):
    image: UploadFile = File()
    audio: UploadFile = File()
    pose: str


def validate_file(file: UploadFile, extensions: list, filetype: str):
    ext = file_ext(file.filename)
    if ext not in extensions:
        raise HTTPException(status_code=400, detail=f"Invalid {filetype} file format")


def validate_image_file(file: UploadFile):
    validate_file(file, [".jpg", ".png"], "image")


def validate_audio_file(file: UploadFile):
    validate_file(file, [".mp3", ".wav"], "audio")


async def save_file(validate: Callable[[UploadFile], None], file: UploadFile) -> str:
    validate(file)
    ext = file_ext(file.filename)
    filename = generate_unique_filename() + ext
    content = await file.read()
    # 确保 INPUT_DIR 以斜杠结尾
    input_path = f"{INPUT_DIR.rstrip('/')}/{filename}"

    try:
        with open(input_path, "wb") as f:
            f.write(content)
        logging.info(f"File saved successfully: {input_path}")
        return input_path
    except IOError as e:
        logging.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500)


async def save_image_file(file: UploadFile) -> str:
    return await save_file(validate_image_file, file)


async def save_audio_file(file: UploadFile) -> str:
    return await save_file(validate_audio_file, file)


@app1.post("/create_video")
async def create_video(
    image: UploadFile = File(), audio: UploadFile = File(), pose: str = "01"
):
    try:
        image_path = await save_image_file(image)
        audio_path = await save_audio_file(audio)

        print(f"image_path: {image_path}")
        print(f"audio_path: {audio_path}")

        output_file = create(image_path, audio_path, pose)

        return {"video": f"/{OUTPUT_DIR.rstrip('/')}/{output_file}"}
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError as e:
        logging.error(f"Permission error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logging.error(f"Unexpected error during video conversion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        delete_file(image_path)
        delete_file(audio_path)


