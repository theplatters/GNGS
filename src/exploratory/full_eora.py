import src.eora as eo
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

eu_countries = [
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
]
eu_countries_plus_ckj = eu_countries + [
    "CHN",  # China
    "KOR",  # Korea
    "JPN",  # Japan
]


eora_true = eo.Eora("data/full_eora")
res = eora_true.q.div(eora_true.x, axis=1)
res
eora_true.q

eora_true.q
eora = eo.test_eora()
eora2 = deepcopy(eora)

eora_true.q.columns[eora_true.q.columns.duplicated()]
eora_true.q[("CHL", "Commodities", "Seafood")]
eora_true.x[("CHL", "Commodities", "Seafood")]
