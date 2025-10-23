import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from src.share_functions import ptt, co2_shares
import json
import pymrio

# Load the built-in world dataset
eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

world = gpd.read_file(url)

plot_dir = "results/plots"
with open("data/north_codes.json") as f:
    north_codes = list(set(json.load(f)))

with open("data/south_codes.json") as f:
    south_codes = list(set(json.load(f)))

with open("data/eu_codes.json") as f:
    eu_codes = list(set(json.load(f)))

# removes Guinnea-Bissau
with open("data/Low_income.json") as f:
    low_income_codes = list(set(json.load(f)))

# removed 'FSM' 'TLS' 'COM' 'KIR' 'SLB'
with open("data/Lower_middle_income.json") as f:
    lower_middle_income_codes = list(set(json.load(f)))

# removes 'TON' 'XKX' 'DMA' 'LCA' 'MHL' 'GNQ' 'GRD' 'VCT' 'TUV'
with open("data/Upper_middle_income.json") as f:
    upper_middle_income_codes = list(set(json.load(f)))

# removed 'MAF' 'IMN' 'SXM' 'CHI' 'NRU' 'MNP' 'GIB' 'FRO' 'KNA' 'PRI' 'TCA' 'CUW' 'ASM' 'GUM' 'VIR' 'PLW'
with open("data/High_income.json") as f:
    high_income_codes = list(set(json.load(f)))


def plot(from_codes, to_codes, to_codes_name):
    non_from_to = [el for el in to_codes if el not in from_codes]
    non_from_to_wo_russia_and_china = [
        el for el in non_from_to if el != "RUS" and el != "CHN" and el != "IND"
    ]

    ptt_df = (
        pd.Series(
            [ptt(eora, from_codes, el) for el in non_from_to],
            index=non_from_to,
        )
        .rename(index={"SUD": "SDN"})
        .rename("ptt")
        .reset_index()
        .rename(columns={"index": "ISO_A3"})
    )
    ptt_df = ptt_df[ptt_df["ptt"] > 1]

    co2_shares_df = (
        co2_shares(eora, from_codes, non_from_to_wo_russia_and_china)
        .sum()
        .div(co2_shares(eora, from_codes, non_from_to).sum().sum())
        .groupby(level=[0])
        .sum()
        .rename("co2_share")
        .reset_index()
        .rename(columns={"region": "ISO_A3"})
    )

    # Merge with world GeoDataFrame
    world_merged = world.merge(ptt_df, on="ISO_A3", how="left").merge(
        co2_shares_df, on="ISO_A3", how="left"
    )

    # Identify outliers
    outliers = world_merged[world_merged["ptt"] > 10]
    non_outliers = world_merged[world_merged["ptt"] <= 10]

    # Plot PTT map
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot normal countries
    world_merged.plot(
        color="lightgray",
        edgecolor="white",  # show country boundaries
        ax=ax,
    )
    outliers.plot(color="red", ax=ax, edgecolor="black")
    non_outliers.plot(
        column="ptt",
        cmap="viridis",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgray"},
    )
    # Plot outliers in red
    plt.title("PTT Map (outliers >10 in red)")
    plt.savefig(plot_dir + "/ptt_map" + to_codes_name + ".png")
    #

    # Plot CO2 shares map (keep original style)
    fig, ax = plt.subplots(figsize=(12, 6))
    world_merged.plot(
        column="co2_share",
        cmap="viridis",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgray"},
    )
    plt.title("CO² shares")
    plt.savefig(plot_dir + "/co2_shares_map" + to_codes_name + ".png")


fc = south_codes + ["CHN"] + ["IND"]
plot(eu_codes, fc, "gs")

world[world["SOV_A3"] == "SDN"]
china_rows = world[world["NAME"].str.contains("Sudan", case=False, na=False)]
print(china_rows[["NAME", "SOV_A3", "ISO_A3", "ADM0_A3"]])
