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

non_eu_south = [el for el in south_codes if el not in eu_codes]
non_eu_south_wo_russia = [el for el in non_eu_south if el != "RUS"]

ptt_df = (
    pd.Series(
        [ptt(eora, eu_codes, el) for el in non_eu_south],
        index=non_eu_south,
    )
    .rename("ptt")
    .sort_values()
    .reset_index()
    .rename(columns={"index": "SOV_A3"})
    .rename(index={"SUD": "SDN"})
)
ptt_df = ptt_df[ptt_df["ptt"] < 10][ptt_df["ptt"] > 1]

co2_shares_df = (
    co2_shares(eora, eu_codes, non_eu_south_wo_russia)
    .sum()
    .rename(index={"SUN": "SDN"})
    .div(co2_shares(eora, eu_codes, non_eu_south).sum().sum())
    .groupby(level=[0])
    .sum()
    .rename("co2_share")
    .reset_index()
    .rename(columns={"region": "SOV_A3"})
)


# Plot the world
fig, ax = plt.subplots(figsize=(12, 6))
# Merge with world GeoDataFrame on SOV_A3
world_merged = world.merge(ptt_df, on="SOV_A3", how="left").merge(
    co2_shares_df, on="SOV_A3", how="left"
)


# Plot
fig, ax = plt.subplots(figsize=(12, 6))
world_merged.plot(
    column="ptt",
    cmap="viridis",
    legend=True,
    ax=ax,
    missing_kwds={"color": "lightgray"},
)  # countries not in your series appear gray
plt.savefig(plot_dir + "/ptt_map.png")
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
world_merged.plot(
    column="co2_share",
    cmap="viridis",
    legend=True,
    ax=ax,
    missing_kwds={"color": "lightgray"},
)  # countries not in your series appear gray
plt.savefig(plot_dir + "/co2_shares_map.png")
plt.show()
