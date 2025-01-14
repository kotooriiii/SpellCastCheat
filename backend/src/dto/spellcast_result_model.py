from pydantic import BaseModel, Field
from typing import List, Tuple

from dto.sequence_model import SequenceModel
from dto.tile_model import TileModel


class SpellCastResultModel(BaseModel):
    sequences: List[SequenceModel] = Field(..., description="List of sequence results sorted by descending value", example=[
        {"cost": 12, "word": "tan", "path": [(0, 0), (1, 1), (2, 2)]},
        {"cost": 3, "word": "no", "path": [(0, 0), (1, 1)]}
    ])

    grid: List[List[TileModel]] = Field(..., description="A 2D array representing the grid of tiles", example=[
        [{"id": 1, "x": 0, "y": 0, "value": "A"}, {"id": 2, "x": 0, "y": 1, "value": "B"}],
        [{"id": 3, "x": 1, "y": 0, "value": "C"}, {"id": 4, "x": 1, "y": 1, "value": "D"}]
    ])

    class Config:
        title = "Spell Cast Result"
        description = "The predicted table and list of sequence results bundled together."
