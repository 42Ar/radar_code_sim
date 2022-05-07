# -*- coding: utf-8 -*-

import json
import numpy as np

def read(name):
    with open(f"{name}.json", "r") as inf_file:
        globals().update(json.load(inf_file))
    global m
    m = np.load(f"{name}.npy")