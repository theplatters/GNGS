import pandas as pd


def embodied_co2_emissions(
    eora, from_codes: list[str] | str, to_codes: list[str]
) -> pd.Series:
    north_demand_from_south = eora.Y.loc[to_codes, from_codes].sum(axis=1)
    north_demand_from_north = eora.Y.loc[from_codes, from_codes].sum(axis=1)

    induced_demand_from_final_demand = eora.L.loc[to_codes, to_codes].dot(
        north_demand_from_south
    )
    induced_demand_from_intermediate_demand = eora.L.loc[to_codes, from_codes].dot(
        north_demand_from_north
    )

    induced_demand = (
        induced_demand_from_final_demand + induced_demand_from_intermediate_demand
    )

    f_shares = eora.Q.S.loc["I-GHG-CO2 emissions", to_codes]

    co2_share = f_shares.mul(induced_demand)
    return co2_share


def embodied_value_added(
    eora,
    from_codes: list[str] | str,
    to_codes: list[str] | str,
    primary_input_label: str = "Compensation of employees D.1",
) -> pd.Series:
    north_demand_from_south = eora.Y.loc[to_codes, from_codes].sum(axis=1)
    north_demand_from_north = eora.Y.loc[from_codes, from_codes].sum(axis=1)

    induced_demand_from_final_demand = eora.L.loc[to_codes, to_codes].dot(
        north_demand_from_south
    )
    induced_demand_from_intermediate_demand = eora.L.loc[to_codes, from_codes].dot(
        north_demand_from_north
    )

    induced_demand = (
        induced_demand_from_final_demand + induced_demand_from_intermediate_demand
    )
    f_shares = eora.VA.S.loc[("Primary input", primary_input_label), to_codes]

    eva = f_shares.mul(induced_demand)
    return eva


def dependency_shares(
    eora,
    from_codes: list[str] | str,
    to_codes: list[str] | str,
    primary_input_label: str = "Compensation of employees D.1",
) -> pd.Series:
    eva = embodied_value_added(eora, from_codes, to_codes, primary_input_label)
    total_value_added = eora.VA.F.loc[
        ("Primary input", primary_input_label), to_codes
    ].sum()

    return eva.div(total_value_added)


def co2_shares(
    eora, from_codes: list[str] | str, to_codes: list[str] | str
) -> pd.Series:
    total_south_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", to_codes].sum().sum()
    return embodied_co2_emissions(eora, from_codes, to_codes).div(total_south_emissions)


def co2_shares_sectoral(eora, from_codes: list[str], to_codes: list[str]) -> pd.Series:
    sectoral_emissions = eora.Q.F.loc["I-GHG-CO2 emissions", to_codes].sum()

    return (
        embodied_co2_emissions(eora, from_codes, to_codes).sum().div(sectoral_emissions)
    )


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
    ft = (
        embodied_co2_emissions(eora, from_codes, to_codes).sum().sum()
        / embodied_value_added(eora, from_codes, to_codes).sum()
    )
    tf = (
        embodied_co2_emissions(eora, to_codes, from_codes).sum().sum()
        / embodied_value_added(eora, to_codes, from_codes).sum()
    )

    return ft / tf
