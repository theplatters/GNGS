import pymrio
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import numpy as np
import geopandas as gpd
import pycountry
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from src.exploratory.co2_shares import ptt

# Create output directories if they don't exist
output_dir = "results"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()

with open("data/north_codes.json") as f:
    north_codes = list(set(json.load(f)))

with open("data/south_codes.json") as f:
    south_codes = list(set(json.load(f)))
    south_codes.remove("SSD")
    south_codes.remove("SDN")

with open("data/eu_codes.json") as f:
    eu_codes = list(set(json.load(f)))

# [Previous functions remain the same...]


def get_country_name(eora_code):
    """Convert Eora country code to full country name"""
    try:
        return eora.meta.regions[eora_code]
    except:
        return eora_code


def get_iso3_code(country_name):
    """Convert country name to ISO3 code"""
    try:
        # Handle special cases
        if country_name == "Viet Nam":
            country_name = "Vietnam"
        elif country_name == "Russian Federation":
            country_name = "Russia"
        elif country_name == "Korea, Republic of":
            country_name = "South Korea"
        elif country_name == "Iran, Islamic Republic of":
            country_name = "Iran"
        elif country_name == "Venezuela, Bolivarian Republic of":
            country_name = "Venezuela"
        elif country_name == "Bolivia (Plurinational State of)":
            country_name = "Bolivia"
        elif country_name == "Tanzania, United Republic of":
            country_name = "Tanzania"
        elif country_name == "Congo, Democratic Republic of the":
            country_name = "Democratic Republic of the Congo"
        elif country_name == "Côte d'Ivoire":
            country_name = "Ivory Coast"
        elif country_name == "Lao People's Democratic Republic":
            country_name = "Laos"
        elif country_name == "Syrian Arab Republic":
            country_name = "Syria"
        elif country_name == "Eswatini":
            country_name = "Swaziland"
        elif country_name == "North Macedonia":
            country_name = "Macedonia"

        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except:
        return None


# Calculate PTT ratios
non_eu_south = [el for el in south_codes if el not in eu_codes]
ptts = pd.Series(
    [ptt(eora, eu_codes, el) for el in non_eu_south],
    index=non_eu_south,
    name="PTT_Ratio",
)

# Create a DataFrame with country information
ptts_df = ptts.reset_index()
ptts_df.columns = ["Eora_Code", "PTT_Ratio"]
ptts_df["Country_Name"] = ptts_df["Eora_Code"].apply(get_country_name)
ptts_df["ISO3"] = ptts_df["Country_Name"].apply(get_iso3_code)

# Drop countries without ISO3 codes
ptts_df = ptts_df.dropna(subset=["ISO3"])

# Load world map data
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Merge PTT data with world map
world_ptt = world.merge(ptts_df, left_on="iso_a3", right_on="ISO3", how="left")

# Create the map visualization
plt.figure(figsize=(20, 12))
ax = plt.gca()

# Plot countries with PTT data
world_ptt_plot = world_ptt.dropna(subset=["PTT_Ratio"])
world_ptt_plot.plot(
    column="PTT_Ratio",
    cmap="RdYlGn_r",  # Red for high PTT, Green for low PTT
    linewidth=0.8,
    edgecolor="0.8",
    legend=True,
    ax=ax,
    legend_kwds={
        "label": "PTT Ratio (EU to Non-EU South)",
        "orientation": "horizontal",
        "shrink": 0.6,
        "pad": 0.05,
        "aspect": 40,
    },
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing data",
        "edgecolor": "0.8",
    },
)

# Highlight EU countries
eu_world = world[
    world["iso_a3"].isin(
        [
            get_iso3_code(get_country_name(code))
            for code in eu_codes
            if get_iso3_code(get_country_name(code)) is not None
        ]
    )
]
eu_world.plot(ax=ax, color="none", edgecolor="blue", linewidth=1.5, alpha=0.7)

# Add country labels for top PTT countries
top_ptt_countries = ptts_df.nlargest(10, "PTT_Ratio")
for idx, row in top_ptt_countries.iterrows():
    country_geom = world_ptt[world_ptt["ISO3"] == row["ISO3"]]
    if not country_geom.empty:
        centroid = country_geom.geometry.centroid.iloc[0]
        ax.text(
            centroid.x,
            centroid.y,
            row["Country_Name"],
            fontsize=8,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

# Add title and formatting
plt.title("PTT Ratios: EU to Non-EU South Countries (2017)", fontsize=16, pad=20)
plt.axis("off")
plt.tight_layout()

# Save the map
plt.savefig(f"{plot_dir}/ptt_world_map.png", dpi=300, bbox_inches="tight")
plt.close()

# Create a regional focus map (Africa, Asia, Latin America)
plt.figure(figsize=(16, 10))
ax = plt.gca()

# Define regions for focus
regions = {
    "Africa": (-20, 55, -35, 40),  # minx, maxx, miny, maxy
    "Asia": (40, 150, -10, 50),
    "Latin America": (-90, -30, -60, 20),
}

# Plot each region separately
for region_name, (minx, maxx, miny, maxy) in regions.items():
    # Filter countries in the region
    region_countries = world_ptt.cx[minx:maxx, miny:maxy]

    # Plot the region
    region_countries.plot(
        column="PTT_Ratio",
        cmap="RdYlGn_r",
        linewidth=0.8,
        edgecolor="0.8",
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "0.8",
        },
    )

    # Add region label
    ax.text(
        (minx + maxx) / 2,
        maxy + 5,
        region_name,
        fontsize=14,
        ha="center",
        weight="bold",
    )

# Add title and formatting
plt.title(
    "PTT Ratios by Region: EU to Non-EU South Countries (2017)", fontsize=16, pad=20
)
plt.axis("off")
plt.tight_layout()

# Save the regional map
plt.savefig(f"{plot_dir}/ptt_regional_map.png", dpi=300, bbox_inches="tight")
plt.close()

# Create a detailed map with top 10 PTT countries highlighted
plt.figure(figsize=(20, 12))
ax = plt.gca()

# Plot all countries with PTT data
world_ptt_plot.plot(
    column="PTT_Ratio",
    cmap="RdYlGn_r",
    linewidth=0.8,
    edgecolor="0.8",
    ax=ax,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "0.8",
    },
)

# Highlight top 10 PTT countries
top_countries = ptts_df.nlargest(10, "PTT_Ratio")
top_geoms = world_ptt[world_ptt["ISO3"].isin(top_countries["ISO3"])]
top_geoms.plot(ax=ax, color="none", edgecolor="red", linewidth=2.5)

# Add labels for top countries
for idx, row in top_countries.iterrows():
    country_geom = world_ptt[world_ptt["ISO3"] == row["ISO3"]]
    if not country_geom.empty:
        centroid = country_geom.geometry.centroid.iloc[0]
        ax.text(
            centroid.x,
            centroid.y,
            f"{row['Country_Name']}\n({row['PTT_Ratio']:.2f})",
            fontsize=9,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"),
        )

# Add EU outline
eu_world.plot(ax=ax, color="none", edgecolor="blue", linewidth=1.5, alpha=0.7)

# Add title and formatting
plt.title("Top 10 PTT Countries Highlighted (EU to Non-EU South)", fontsize=16, pad=20)
plt.axis("off")
plt.tight_layout()

# Save the detailed map
plt.savefig(f"{plot_dir}/ptt_top_countries_map.png", dpi=300, bbox_inches="tight")
plt.close()

# Save the PTT data with ISO codes for reference
ptts_df.to_csv(f"{output_dir}/ptt_ratios_with_geocodes.csv", index=False)

print(f"PTT world map saved to '{plot_dir}/ptt_world_map.png'")
print(f"PTT regional map saved to '{plot_dir}/ptt_regional_map.png'")
print(f"PTT top countries map saved to '{plot_dir}/ptt_top_countries_map.png'")
print(f"PTT data with geocodes saved to '{output_dir}/ptt_ratios_with_geocodes.csv'")
