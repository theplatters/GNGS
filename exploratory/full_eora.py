from eora_loader import EoraLoader
import difflib
import pandas as pd

eu_countries_plus_ckj = [
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
    "CHN",  # China
    "KOR",  # Korea
    "JPN",  # Japan
]

eora = EoraLoader("data/full_eora")


t = eora.t[eu_countries_plus_ckj].copy()


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

sectors = sector_country_matrix.index.tolist()
similar_sectors = []
threshold = 0.9  # 90% similarity

for i in range(len(sectors)):
    for j in range(i + 1, len(sectors)):
        s1, s2 = sectors[i], sectors[j]
        score = difflib.SequenceMatcher(None, s1, s2).ratio()
        if score >= threshold:
            similar_sectors.append((s1, s2, round(score, 3)))

similar_df = pd.DataFrame(
    similar_sectors, columns=["Sector 1", "Sector 2", "Similarity"]
)
similar_df.sort_values(by="Similarity", ascending=False, inplace=True)

print(similar_df.head(20))

