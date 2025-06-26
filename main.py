import matplotlib.pyplot as plt
import seaborn as sns
import pymrio
import pandas as pd  # Good practice for type hinting and direct use if needed
import country_converter as coco
import numpy as np

years = range(2010, 2018)


def analyze_trade(years):
    exports_into_eu = {}
    imports_from_eu = {}
    exports_into_china = {}
    imports_from_china = {}

    for year in years:
        eora = pymrio.parse_eora26(year=year, path="data/" + str(year)).calc_all()
        eora.aggregate(
            region_agg=coco.agg_conc(
                original_countries="Eora",
                aggregates=[
                    {"CHL": "LE", "BOL": "LE", "ARG": "LE"},
                    {"CHN": "China"},
                    "EU",
                ],
                missing_countries="Other",
                merge_multiple_string=None,
            )
        )

        exports_into_eu[year] = eora.Z.loc[
            ("LE", "Mining and Quarrying"),
            ("EU", slice(None)),
        ].sum()

        imports_from_eu[year] = eora.Z.loc[
            ("EU", "Mining and Quarrying"),
            ("LE", slice(None)),
        ].sum()

        exports_into_china[year] = eora.Z.loc[
            ("LE", "Mining and Quarrying"),
            ("China", slice(None)),
        ].sum()

        imports_from_china[year] = eora.Z.loc[
            ("China", "Mining and Quarrying"),
            ("LE", slice(None)),
        ].sum()

    plt.plot(years, [exports_into_eu[year] for year in years], label="Exports into EU")
    plt.plot(years, [imports_from_eu[year] for year in years], label="Imports from EU")
    plt.plot(
        years, [exports_into_china[year] for year in years], label="Exports into China"
    )
    plt.plot(
        years, [imports_from_china[year] for year in years], label="Imports from China"
    )
    plt.xlabel("Year")
    plt.ylabel("Trade Value")
    plt.title("Trade with EU (2000–2017)")
    plt.legend()
    plt.grid(True)
    plt.savefig("import-export-lithium-exporting-countries.png")


analyze_trade(years)
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


mining_data = {}
top_mining = {}
significant_data = {}
for year in years:
    eora = pymrio.parse_eora26(year=year, path="data/" + str(year)).calc_all()

    mining_data[year] = (
        eora.Z[eu_countries]
        .T.groupby(level=1)
        .sum()
        .T.xs("Mining and Quarrying", level="sector")
    )
    top_mining[year] = (
        mining_data[year].sum(axis=1).sort_values(ascending=False).head(10)
    )

    # Calculate row and column sums
    row_sums = mining_data[year].sum(axis=1)
    col_sums = mining_data[year].sum(axis=0)

    # Set significance thresholds (adjust these as needed)
    row_threshold = row_sums.quantile(0.75)  # Top 25% of countries
    col_threshold = col_sums.quantile(0.75)  # Top 25% of sectors
    significant_data[year] = mining_data[year].loc[
        row_sums >= row_threshold, col_sums >= col_threshold
    ]

significant_data
# Create the filtered heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    significant_data[2017],
    cmap="rocket_r",  # Reverse rocket colormap for better visibility
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    cbar_kws={"label": "Economic Value"},
    annot_kws={"size": 8},
)

plt.title("Significant Contributors from Mining and Quarrying Sector", pad=20)
plt.xlabel("Contributing to Industries (Top 25% by contribution)")
plt.ylabel("Countries (Top 25% by total value)")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

top_countries = (
    mining_data[2017].sum(axis=1).sort_values(ascending=False).head(15).index
)

plt.figure(figsize=(14, 7))
mining_data[2017].loc[top_countries].plot(
    kind="bar", stacked=True, colormap="viridis", edgecolor="black", linewidth=0.5
)

plt.title(
    "Breakdown of Mining and Quarrying Economic Activity by Sector (Top 10 Countries)"
)
plt.ylabel("Total Value")
plt.xlabel("Country")
plt.legend(title="Sectors", bbox_to_anchor=(1.05, 1))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# ==========================================================================
# ==========================================================================
# ==========================================================================
test_mrio = pymrio.load_test()
eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()
eora.find("lithium")

eora.Q.D_cba.loc[
    "Raw material inputs, itemized", "A.2.2.9 - Other metal ores - gross ore"
]
list(filter(lambda x: x.startswith("Raw"), eora.Q.M.index.get_level_values(0).unique()))
metal_ore_inputs = eora.Q.D_pba.loc["Raw material inputs, itemized"].loc[
    lambda df: df.index.str.startswith("A.2.2")
]

# Extract Mining and Quarrying columns only
metal_ore_in_mining = metal_ore_inputs.loc[:, (slice(None), "Mining and Quarrying")]

# Compute ratio of 'Other metal ores' over total metal ores
filtered_ratio = (
    metal_ore_in_mining.loc["A.2.2.9 - Other metal ores - gross ore"]
    / metal_ore_in_mining.sum()
)

gross_trade = eora.get_gross_trade()
gross_trade.bilat_flows.head()
filtered_ratio["CHL"]
eora.Q.D_pba.loc[
    ("Raw material inputs, itemized", "A.2.2.9 - Other metal ores - gross ore"), "NLD"
][lambda x: x > 0]

metal_ore_inputs["CHL"].loc[(metal_ore_inputs > 0.0).any(axis=1)]

metal_ore_inputs["CHL"][metal_ore_inputs["CHL"] > 0].stack()
metal_ore_inputs["BOL"][metal_ore_inputs["BOL"] > 0].stack()
metal_ore_inputs["CHN"][metal_ore_inputs["CHN"] > 0].stack()
metal_ore_inputs["AUS"][metal_ore_inputs["AUS"] > 0].stack()

# Calculate metal ore inputs using consumption-based accounts for EU countries
#
eora.Q.D_pba.loc["Raw material inputs, itemized"].loc[
    lambda df: df.index.str.startswith("A.3")
]["CHL"][lambda x: x > 0].stack()

eora.Q.D_pba.loc["Raw material inputs, itemized"].loc[
    lambda df: df.index.str.startswith("A.3")
]["CHN"][lambda x: x > 0].stack()

metal_ore_inputs["CHL"].loc[(metal_ore_inputs > 0.0).any(axis=1)]

metal_ore_inputs["CHL"][metal_ore_inputs["CHL"] > 0].stack()
metal_ore_inputs["BOL"][metal_ore_inputs["BOL"] > 0].stack()
metal_ore_inputs["CHN"][metal_ore_inputs["CHN"] > 0].stack()
metal_ore_inputs["AUS"][metal_ore_inputs["AUS"] > 0].stack()
