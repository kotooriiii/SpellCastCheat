from pydantic import BaseModel, Field


class TileModel(BaseModel):
    letter: str = Field(..., min_length=1, max_length=1, pattern="^[A-Za-z]$",
                        description="A single alphabetic character.")
    number: int = Field(..., ge=0, description="A non-negative integer representing the tile's number.")
    powerup: bool = Field(..., description="Indicates whether the tile has a powerup.")
    double_letter: bool = Field(..., description="Indicates whether the tile has a double letter bonus.")
    triple_letter: bool = Field(..., description="Indicates whether the tile has a triple letter bonus.")
    double_point: bool = Field(..., description="Indicates whether the tile has a double point bonus.")

    class Config:
        title = "Tile"
        description = "Model representing a single tile on the grid"
