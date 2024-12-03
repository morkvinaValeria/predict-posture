
import os
from fastapi import APIRouter
from keras.models import load_model
from ..schemas.data_to_predict import BackViewDataToPredict, SideViewDataToPredict
from ..schemas.lang import Lang
from ..schemas.prediction import BackViewPrediction, SideViewPrediction
from ..utils.utils import generate_message

router = APIRouter()

@router.post("/predict-posture")
def posture_points(data: BackViewDataToPredict | SideViewDataToPredict, lang: Lang = Lang.en) -> BackViewPrediction | SideViewPrediction:
    BACK_LABELS={'left_c_scoliotic_posture': 0, 'neutral_posture': 1, 'right_c_scoliotic_posture': 2, 's_scoliotic_posture': 3}
    SIDE_LABELS={'kyphotic_lordotic_posture': 0, 'kyphotic_posture': 1, 'lordotic_posture': 2, 'neutral_posture': 3}
    
    model_name = 'side' if data.sideView == True else 'back'
    model = load_model(os.path.join('trained-model', f'posture_{model_name}_view_assessment_trained_model.h5'))
    labels = SIDE_LABELS if data.sideView == True else BACK_LABELS
    
    preds = model.predict([data.angles])
    preds = preds.tolist()
    res = {}
    for i in range(0, len(preds[0])):
        res.update({list(labels.keys())
                    [list(labels.values()).index(i)]: preds[0][i]})
    message = generate_message(res, lang)
    res.update({"message": message})
    return res
