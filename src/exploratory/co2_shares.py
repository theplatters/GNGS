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
    south_codes.remove("SSD")
    south_codes.remove("SDN")
Y_sn = eora.Y.loc[south_codes, north_codes].sum(axis=1)
Z_sn = eora.Z.loc[south_codes, north_codes].sum(axis=1)
demand_sn = Y_sn + Z_sn

south_leontief = eora.L.loc[south_codes, south_codes]

eora.Q.F.loc["I-GHG-CO2 emissions", south_codes].sum(axis=0).div(
    eora.x.loc[south_codes]
)

f_shares = eora.Q.S.loc["I-GHG-CO2 emissions", south_codes].sum(axis=0)
total_south_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", south_codes].sum().sum()

total = f_shares.mul(south_leontief.dot(Y_sn + Z_sn)).div(total_south_emissions)
final_demand = f_shares.mul(south_leontief.dot(Y_sn)).div(total_south_emissions)
intermediates = f_shares.mul(south_leontief.dot(Z_sn)).div(total_south_emissions)

total.describe()
total.sum()

# Emission intensities (factor production coefficients)
f_south = eora.Q.S.loc["I-GHG-CO2 emissions", south_codes]

# Output required in each Southern sector to satisfy Northern demand
req_total = south_leontief.dot(Y_sn + Z_sn)

# Sectoral emissions caused by North demand
emissions_from_north_sectoral = f_south * req_total

# Total emissions of each Southern sector
sector_total_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", south_codes]
# Share per sector (this is what you want)
sectoral_share = emissions_from_north_sectoral / sector_total_emissions

print(sectoral_share.sum().describe())
