from pydantic import BaseModel, Field
from typing import List, Tuple


class SequenceModel(BaseModel):
    cost: int = Field(..., description="The total cost associated with forming this word", example=12)
    word: str = Field(..., description="The word that was formed", example="elephant")
    path: List[Tuple[int, int]] = Field(..., description="The path of coordinates taken to form the word on the grid",
                                        example=[(0, 0), (1, 1)])

    class Config:
        title = "Sequence"
        description = "Represents a word formed in a spell casting, including its cost and the path taken on the grid."
