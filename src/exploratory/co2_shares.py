import pymrio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()


with open("data/north_codes.json") as f:
    north_codes = list(set(json.load(f)))

with open("data/south_codes.json") as f:
    south_codes = list(set(json.load(f)))

eora.Y.loc[list(south_codes), list(north_codes)].sum(axis=1)
