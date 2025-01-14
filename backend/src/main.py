from fastapi import FastAPI, File, UploadFile, APIRouter

from typing import List, Dict
from io import BytesIO
from PIL import Image

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware  # Turns out it doesn't matter where you import this from

import logic.driver
from dto.spellcast_result_model import SpellCastResultModel
from logic.word_finder import WordFinder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def initialize_shared_data_once():
    """Initialize shared data for WordFinder on app startup."""
    WordFinder.initialize_shared_data()


# Endpoint to handle the image and return 2D array of tiles + predictions
@app.post("/process_image", response_model=SpellCastResultModel)
async def process_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Check if the image is JPEG
    if image.format == "JPEG":
        # Convert JPEG to PNG in memory
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image = Image.open(buffer)

    return logic.driver.process_image(image)
