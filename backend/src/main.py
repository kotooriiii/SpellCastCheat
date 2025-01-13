from fastapi import FastAPI, File, UploadFile, APIRouter

from typing import List, Dict
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import Tuple

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware  # Turns out it doesn't matter where you import this from

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Tile Pydantic model for serialization
class TileModel(BaseModel):
    id: int
    x: int
    y: int
    value: str

    class Config:
        orm_mode = True


# Prediction model
class Prediction(BaseModel):
    cost: float
    word: str
    path: List[Tuple[int, int]]


# Tile class (not using Pydantic here, since FastAPI will use the Pydantic model)
class Tile:
    def __init__(self, id: int, x: int, y: int, value: str):
        self.id = id
        self.x = x
        self.y = y
        self.value = value

    def __repr__(self):
        return f"Tile(id={self.id}, x={self.x}, y={self.y}, value='{self.value}')"


# Mock function to simulate ML predictions
def get_predictions(image: Image):
    # Here you would call your ML model to process the image
    # For now, returning mock predictions
    return [
        {"cost": 0.95, "word": "example", "path": [(0, 0), (1, 1)]},
        {"cost": 0.89, "word": "test", "path": [(1, 1), (2, 2)]},
    ]


# Endpoint to handle the image and return 2D array of tiles + predictions
@app.post("/process_image", response_model=Dict[str, List])
async def process_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Get mock predictions
    predictions = get_predictions(image)

    # Generate a 2D array of Tiles (for this example, 3x3 grid)
    tile_grid = [
        [Tile(id=i + j * 3, x=j, y=i, value=f"Tile {i}-{j}") for i in range(3)]
        for j in range(3)
    ]

    # Convert the tile grid to a list of TileModel instances for response
    tile_grid_serialized = [
        [TileModel(id=tile.id, x=tile.x, y=tile.y, value=tile.value) for tile in row]
        for row in tile_grid
    ]

    return {"tiles": tile_grid_serialized, "predictions": predictions}