from pydantic import BaseModel, Field


class BackViewDataToPredict(BaseModel):
    sideView: bool = False
    angles: list[float] = Field(min_items=3, max_items=3)

class SideViewDataToPredict(BaseModel):
    sideView: bool = True
    angles: list[float] = Field(min_items=4, max_items=4)
    
