from src.samples.predictions import predictions
from src.utils.constants import MOST_PROBABLY, PROBABLY, SMALL_CHANCE, ACTION

def generate_message(dict, lang):
    message = ''
    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    is_added = False
    for key in sorted_dict.keys():
        key_val = predictions[key]
        descr = key_val[f"description_{lang}"]
        act = key_val[f"action_{lang}"]
        if dict[key] >= 0.8:
            if is_added:
                message += '\n'
            message += f'{MOST_PROBABLY} {descr} {ACTION}: {act}'
            is_added = True
        elif dict[key] >= 0.35 and dict[key] < 0.8:
            if is_added:
                message += '\n'
            message += f'{PROBABLY} {descr} {ACTION}: {act}'
            is_added = True
        elif dict[key] >= 0.2 and dict[key] < 0.35:
            if is_added:
                message += '\n'
            message += f'{SMALL_CHANCE} {descr} {ACTION}: {act}'
            is_added = True
    return message        