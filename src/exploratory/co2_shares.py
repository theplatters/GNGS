import pymrio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymrio.core.mriosystem import IOSystem
import json

eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()


with open("data/north_codes.json") as f:
    north_codes = list(set(json.load(f)))

with open("data/south_codes.json") as f:
    south_codes = list(set(json.load(f)))
    south_codes.remove("SSD")
    south_codes.remove("SDN")


def co2_shares(eora: IOSystem, from_codes: list[str], to_codes: list[str]) -> pd.Series:
    Y_sn = eora.Y.loc[to_codes, from_codes].sum(axis=1)
    Z_sn = eora.Z.loc[to_codes, from_codes].sum(axis=1)
    demand_sn = Y_sn + Z_sn

    south_leontief = eora.L.loc[to_codes, to_codes]

    f_shares = eora.Q.S.loc["I-GHG-CO2 emissions", to_codes].sum(axis=0)
    total_south_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", to_codes].sum().sum()

    co2_share = f_shares.mul(south_leontief.dot(demand_sn)).div(total_south_emissions)
    return co2_share


def co2_shares_sectoral(
    eora: IOSystem, from_codes: list[str], to_codes: list[str]
) -> pd.Series:
    Y_sn = eora.Y.loc[to_codes, from_codes].sum(axis=1)
    Z_sn = eora.Z.loc[to_codes, from_codes].sum(axis=1)
    demand_sn = Y_sn + Z_sn

    south_leontief = eora.L.loc[to_codes, to_codes]

    f_shares = eora.Q.S.loc["I-GHG-CO2 emissions", to_codes].sum(axis=0)
    total_south_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", to_codes].sum(axis=0)

    co2_share = f_shares.mul(south_leontief.dot(demand_sn)).div(total_south_emissions)
    return co2_share


def co2_total(eora: IOSystem, from_codes: list[str], to_codes: list[str]) -> pd.Series:
    emissions = {}
    indices: pd.MultiIndex = eora.A.index[
        eora.A.index._get_level_values(0).isin(from_codes)
    ]

    f_shares = eora.Q.S.loc["I-GHG-CO2 emissions"].sum(axis=0)
    for sec in from_codes:
        Y_sn = eora.Y.loc[to_codes, sec].sum(axis=1)
        Y_total = eora.Y.loc[:, sec].sum(axis=1)

        co2_total = f_shares.mul(eora.L.dot(Y_total)).sum()
        co2_south = (
            f_shares[to_codes].mul(eora.L.loc[to_codes, to_codes].dot(Y_sn)).sum()
        )
        emissions[(sec, "Final Demand")] = co2_south / co2_total

    for sec in indices:
        Z_sn = eora.Z.loc[to_codes, sec]
        Z_total = eora.Z.loc[:, sec]
        co2_total = f_shares.mul(eora.L.dot(Z_total)).sum()
        co2_south = (
            f_shares[to_codes].mul(eora.L.loc[to_codes, to_codes].dot(Z_sn)).sum()
        )
        emissions[sec] = co2_south / co2_total

    res = pd.Series(emissions)
    res.index.names = ["Region", "Sector"]
    return res


res3 = co2_total(eora, north_codes, south_codes)
co2_shares(eora, north_codes, south_codes)
co2_shares_sectoral(eora, north_codes, south_codes).nsmallest(50)
co2_shares_sectoral(eora, north_codes, south_codes).nsmallest(50)

eora.Q.F.loc["I-GHG-CO2 emissions", south_codes].sum(axis=0).div(
    eora.x.loc[south_codes, "indout"]
)
eora.Y.loc[south_codes, "AUT"].sum(axis=1)
eora.L.loc[south_codes, south_co]
