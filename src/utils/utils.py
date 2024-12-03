from src.samples.predictions import predictions
from src.utils.constants import MOST_PROBABLY_EN, PROBABLY_EN, SMALL_CHANCE_EN, ACTION_EN, MOST_PROBABLY_UA, PROBABLY_UA, SMALL_CHANCE_UA, ACTION_UA
from ..schemas.lang import Lang

def generate_message(dict, lang = Lang.en):
    message = ''
    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    is_added = False
    needed_lang = Lang.en.value if lang == Lang.en else Lang.ua.value
    for key in sorted_dict.keys():
        key_val = predictions[key]
        descr = key_val[f"description_{needed_lang}"]
        act = key_val[f"action_{needed_lang}"]
        if dict[key] >= 0.8:
            if is_added:
                message += '\n'
            message += f'{MOST_PROBABLY_EN if lang == Lang.en else MOST_PROBABLY_UA} {descr} {ACTION_EN if lang == Lang.en else ACTION_UA}: {act}'
            is_added = True
        elif dict[key] >= 0.35 and dict[key] < 0.8:
            if is_added:
                message += '\n'
            message += f'{PROBABLY_EN if lang == Lang.en else PROBABLY_UA} {descr} {ACTION_EN if lang == Lang.en else ACTION_UA}: {act}'
            is_added = True
        elif dict[key] >= 0.25 and dict[key] < 0.35:
            if is_added:
                message += '\n'
            message += f'{SMALL_CHANCE_EN if lang == Lang.en else SMALL_CHANCE_UA} {descr} {ACTION_EN if lang == Lang.en else ACTION_UA}: {act}'
            is_added = True
    return message        