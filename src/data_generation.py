import pymrio
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import time

from src.share_functions import (
    ptt,
    co2_shares,
    embodied_value_added,
    embodied_co2_emissions,
    dependency_shares,
    co2_shares_sectoral,
    co2_total,
)

# Create output directories if they don't exist
output_dir = "results"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


start = time.time()
eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()
end = time.time()
print(end - start)

with open("data/north_codes.json") as f:
    north_codes = list(set(json.load(f)))

with open("data/south_codes.json") as f:
    south_codes = list(set(json.load(f)))

with open("data/eu_codes.json") as f:
    eu_codes = list(set(json.load(f)))

# Calculate interesting results
non_eu_south = [el for el in south_codes if el not in eu_codes]
non_eu_south_wo_russia = [el for el in non_eu_south if el != "RUS"]

# 1. Total CO2 emissions
total_co2 = co2_total(eora, eu_codes, non_eu_south)
total_co2.name = "Share of Indicrect CO2 produced in the global south"

total_co2_wo_russia = co2_total(eora, eu_codes, non_eu_south_wo_russia)
total_co2_wo_russia.name = (
    "Share of Indicrect CO2 produced in the global south without russia"
)

merged_co2 = pd.DataFrame(
    {"With Russia": total_co2, "Without Russia": total_co2_wo_russia}
)


merged_co2.to_csv(output_dir + "/share_produced_by_gs.csv")
# 4. Dependency shares
dep_shares = dependency_shares(eora, north_codes, south_codes).sum()

# 6. PTT ratios
ptts = pd.Series(
    [ptt(eora, eu_codes, el) for el in non_eu_south],
    index=non_eu_south,
    name="PTT_Ratio",
)


# Additional comprehensive results
# All CO2 shares
all_co2_shares = co2_shares(eora, eu_codes, non_eu_south).sum()
all_co2_shares_wo_russia = co2_shares(eora, eu_codes, non_eu_south_wo_russia).sum()
all_co2_shares.name = "CO2 shares"
all_co2_shares_wo_russia.name = "CO2 shares without russia"


all_embodied_co2 = embodied_co2_emissions(eora, eu_codes, non_eu_south).sum()
all_embodied_co2.name = "Embodied CO2"
all_embodied_value_added = embodied_value_added(eora, eu_codes, non_eu_south)
all_embodied_value_added.name = "Embodied value added"

# All sectoral CO2 shares
all_sectoral_shares = co2_shares_sectoral(eora, eu_codes, non_eu_south)
all_sectoral_shares.name = "Sectoral CO2 share"

res = pd.concat(
    [all_co2_shares, all_sectoral_shares, all_embodied_co2, all_embodied_value_added],
    axis=1,
)
res.to_csv(output_dir + "/co2_res.csv")
ptts.to_csv(output_dir + "/ptts.csv")

print(f"All results saved to '{output_dir}' directory")
