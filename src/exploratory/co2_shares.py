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


def embodied_co2_emissions(
    eora, from_codes: list[str] | str, to_codes: list[str]
) -> pd.Series:
    Y_sn = eora.Y.loc[to_codes, from_codes].sum(axis=1)
    Z_sn = eora.Z.loc[to_codes, from_codes].sum(axis=1)
    demand_sn = Y_sn + Z_sn

    south_leontief = eora.L.loc[to_codes, to_codes]

    f_shares = eora.Q.S.loc["I-GHG-CO2 emissions", to_codes].sum(axis=0)

    co2_share = f_shares.mul(south_leontief.dot(demand_sn))
    return co2_share


def embodied_value_added(
    eora,
    from_codes: list[str] | str,
    to_codes: list[str] | str,
    primary_input_label: str = "Compensation of employees D.1",
) -> pd.Series:
    Y_sn = eora.Y.loc[to_codes, from_codes].sum(axis=1)
    Z_sn = eora.Z.loc[to_codes, from_codes].sum(axis=1)
    demand_sn = Y_sn + Z_sn

    south_leontief = eora.L.loc[to_codes, to_codes]

    f_shares = eora.VA.S.loc[("Primary input", primary_input_label), to_codes]

    eva = f_shares.mul(south_leontief.dot(demand_sn))
    return eva


def co2_shares(
    eora, from_codes: list[str] | str, to_codes: list[str] | str
) -> pd.Series:
    total_south_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", to_codes].sum().sum()
    return embodied_co2_emissions(eora, from_codes, to_codes).div(total_south_emissions)


def co2_shares_sectoral(eora, from_codes: list[str], to_codes: list[str]) -> pd.Series:
    sectoral_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", to_codes].sum(axis=0)

    return embodied_co2_emissions(eora, from_codes, to_codes).div(sectoral_emissions)


def co2_total(
    eora, from_codes: list[str] | str, to_codes: list[str] | str
) -> pd.Series:
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


def ptt(eora, from_codes: list[str] | str, to_codes: list[str] | str) -> pd.DataFrame:
    """
    Calculates the ratio of embodied CO2 emissions to embodied value added
    for each specified trade flow between regions.
    """
    if eora is None:
        print("Eora data not loaded. Skipping PTT calculation.")
        return {}

    indices: pd.MultiIndex = eora.A.index[
        eora.A.index._get_level_values(0).isin(from_codes)
    ]
    columns: pd.MultiIndex = eora.A.index[
        eora.A.index._get_level_values(0).isin(to_codes)
    ]

    ft = embodied_co2_emissions(eora, from_codes, to_codes) / embodied_value_added(
        eora, from_codes, to_codes
    )
    tf = embodied_co2_emissions(eora, to_codes, from_codes) / embodied_value_added(
        eora, to_codes, from_codes
    )

    res = tf.values[:, None] / ft.values
    res = pd.DataFrame(res, index=tf.index, columns=ft.index)
    res.replace([float("inf"), -float("inf")], np.nan, inplace=True)
    return res


co2_total(eora, north_codes, south_codes)
co2_shares(eora, north_codes, south_codes).sum()
co2_shares_sectoral(eora, north_codes, south_codes)


res = ptt(eora, north_codes, south_codes)

res2 = ptt(eora, south_codes, north_codes)


res.mean(skipna=True).mean(skipna=True)
res2.mean(skipna=True).mean(skipna=True)
res.loc["AUT"]
