from pydantic import BaseModel


class BackViewPrediction(BaseModel):
    message: str
    neutral_posture: float
    left_c_scoliotic_posture: float
    right_c_scoliotic_posture: float
    s_scoliotic_posture: float

class SideViewPrediction(BaseModel):
    message: str
    neutral_posture: float
    kyphotic_lordotic_posture: float
    kyphotic_posture: float
    lordotic_posture: float
