from eora import Eora
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

eu_countries = [
    "AUT",  # Austria
    "BEL",  # Belgium
    "BGR",  # Bulgaria
    "HRV",  # Croatia
    "CYP",  # Cyprus
    "CZE",  # Czechia
    "DNK",  # Denmark
    "EST",  # Estonia
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "GRC",  # Greece
    "HUN",  # Hungary
    "IRL",  # Ireland
    "ITA",  # Italy
    "LVA",  # Latvia
    "LTU",  # Lithuania
    "LUX",  # Luxembourg
    "MLT",  # Malta
    "NLD",  # Netherlands
    "POL",  # Poland
    "PRT",  # Portugal
    "ROU",  # Romania
    "SVK",  # Slovakia
    "SVN",  # Slovenia
    "ESP",  # Spain
    "SWE",  # Sweden
]
eu_countries_plus_ckj = eu_countries + [
    "CHN",  # China
    "KOR",  # Korea
    "JPN",  # Japan
]

eora = Eora("data/full_eora")


# Step 1: Extract MultiIndex as DataFrame
col_index = pd.DataFrame(t.columns.tolist(), columns=["CountryA3", "Entity", "Sector"])

# Step 2: Drop duplicates
unique_pairs = col_index[["Sector", "CountryA3"]].drop_duplicates()

# Step 3: Create presence matrix (Sector as index, Country as columns)
sector_country_matrix = (
    unique_pairs.assign(present=1)
    .pivot(index="Sector", columns="CountryA3", values="present")
    .fillna(0)
    .astype(int)
)

# Optional: sort rows and columns for readability
sector_country_matrix = sector_country_matrix.sort_index().sort_index(axis=1)

# Step 4: Save to CSV
sector_country_matrix.to_csv("sector_by_country_matrix.csv")


country_car_sector = {
    "Passenger motor cars": ["JPN"],
    "Motor vehicles": ["CHN", "ESP"],
    "Sale of motor vehicles, motorcycles etc.": ["DNK"],
    "Manufacture of motor vehicles etc.": ["DNK"],
    "Passenger cars and parts": ["DEU"],
    "Motor vehicles, trailers and semi-trailers": [
        "AUT",
        "BEL",
        "CZE",
        "EST",
        "FIN",
        "FRA",
        "GRC",
        "HUN",
        "IRL",
        "LTU",
        "LVA",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "SWE",
        "ITA",
    ],
    "Motor vehicles and parts": ["KOR"],
}

# missing BGR HVR


sector_rename_map = {
    "Passenger motor cars": "car",
    "Motor vehicles": "car",
    "Manufacture of motor vehicles etc.": "car",
    "Passenger cars and parts": "car",
    "Motor vehicles, trailers and semi-trailers": "car",
    "Motor vehicles and parts": "car",
}

t = eora.t.rename(index=sector_rename_map, columns=sector_rename_map, level="Sector")
a = eora.a.rename(index=sector_rename_map, columns=sector_rename_map, level="Sector")
y = eora.y.rename(index=sector_rename_map, level="Sector")
l = eora.l.rename(index=sector_rename_map, columns=sector_rename_map, level="Sector")
car_leontief = l[eu_countries_plus_ckj].xs(key="car", axis=1, level="Sector")
final_demand = (
    y.loc[eu_countries_plus_ckj].xs(key="car", level="Sector")[eu_countries].sum(axis=1)
)
res = pd.DataFrame(
    np.dot(car_leontief, final_demand),
    index=car_leontief.index,
    columns=["car_imports"],
)

t

res.nlargest(50, columns="car_imports").plot(kind="bar", legend=False)
plt.ylabel("Car Imports")
plt.title("Car Imports by Index")
plt.tight_layout()
plt.show()


sorted_a = (
    a[eu_countries_plus_ckj]
    .xs(key="car", axis=1, level="Sector")
    .mean(axis=1)
    .sort_values(ascending=False)
)
sorted_a.to_csv("sorted_a.csv")
