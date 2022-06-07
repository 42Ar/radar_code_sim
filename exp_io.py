# -*- coding: utf-8 -*-
import numpy as np
import json


def load(exp):
    exp_data = np.load(f"preprocessed/{exp}.npy")
    exp_info = json.load(open(f"preprocessed/{exp}.json"))
    return exp_data, exp_info


def save(exp_info, exp_data):
    exp = f"{exp_info['year']}_{exp_info['month']}_{exp_info['day']}_{exp_info['hour']}_{exp_info['minute']}_{exp_info['second']}_{exp_info['index']}"
    np.save(f"preprocessed/{exp}.npy", exp_data)
    json.dump(exp_info, open(f"preprocessed/{exp}.json", "w"), indent=4)

