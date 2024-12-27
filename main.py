import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from utils import file_ext, generate_unique_filename
from config import INPUT_DIR

app = FastAPI(root_path="/api")
app1 = FastAPI()

app.mount("/v1", app1)


class CreateVideoRequest(BaseModel):
    image: UploadFile
    audio: UploadFile


def validate_image_file(file: UploadFile):
    valid_extensions = [".jpg", ".png"]
    ext = file_ext(file.filename)
    if ext not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid image file format")


def validate_audio_file(file: UploadFile):
    valid_extensions = [".mp3", ".wav"]
    ext = file_ext(file.filename)
    if ext not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid audio file format")


@app.post("/")
async def create_video(request: CreateVideoRequest):
    try:
        validate_image_file(request.image)
        filename = generate_unique_filename() + ext
        content = await request.image.read()
        # 确保 INPUT_DIR 以斜杠结尾
        input_path = f"{INPUT_DIR.rstrip('/')}/{filename}"
        with open(input_path, "wb") as f:
            f.write(content)
        output_file = generate_chroma_key_video(filename)
        return {"report": f"/output/{output_file}"}

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError as e:
        logging.error(f"Permission error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logging.error(f"Unexpected error during video conversion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
