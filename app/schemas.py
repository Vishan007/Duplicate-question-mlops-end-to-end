from typing import List
from fastapi import Query
from pydantic import BaseModel ,validator

class Text(BaseModel):
    q1: str = Query(None, min_length=1)
    q2: str = Query(None, min_length=1)

    @validator("q1")
    def q1_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value
    
    @validator("q2")
    def q2_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value
    
class PredictPayload(BaseModel):
    texts: List[Text]