import matplotlib.pyplot as plt
import seaborn as sns
import pymrio
import pandas as pd  # Good practice for type hinting and direct use if needed
import country_converter as coco
import numpy as np

eora = pymrio.parse_eora26(year=2017, path="data/" + str(2017)).calc_all()


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

north_codes = {
    "AUS",
    "AUT",
    "BHR",
    "BEL",
    "VGB",
    "BRN",
    "CAN",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "ISL",
    "IRL",
    "ISR",
    "ITA",
    "JPN",
    "KWT",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "NZL",
    "NOR",
    "OMN",
    "PRT",
    "QAT",
    "SAU",
    "SGP",
    "SVK",
    "SVN",
    "KOR",
    "ESP",
    "SWE",
    "CHE",
    "TTO",
    "ARE",
    "GBR",
    "USA",
}

south_codes = [
    # Low income (LI)
    "AFG",
    "BGD",
    "BEN",
    "BFA",
    "BDI",
    "KHM",
    "CMR",
    "CAF",
    "TCD",
    "CIV",
    "DJI",
    "COD",
    "ERI",
    "ETH",
    "GMB",
    "GHA",
    "GIN",
    "HTI",
    "HND",
    "KEN",
    "KGZ",
    "LAO",
    "LSO",
    "LBR",
    "MDG",
    "MWI",
    "MLI",
    "MRT",
    "MOZ",
    "MMR",
    "NPL",
    "NIC",
    "NER",
    "PRK",
    "RWA",
    "STP",
    "SEN",
    "SLE",
    "SOM",
    "SYR",
    "TJK",
    "TZA",
    "TGO",
    "UGA",
    "VUT",
    "YEM",
    "ZMB",
    "ZWE",
    # Lower-middle income (LMI)
    "ALB",
    "AGO",
    "ARM",
    "BLZ",
    "BTN",
    "BOL",
    "BIH",
    "CPV",
    "COG",
    "ECU",
    "EGY",
    "SLV",
    "FJI",
    "GEO",
    "GTM",
    "IDN",
    "JAM",
    "JOR",
    "MDV",
    "MDA",
    "MNG",
    "MAR",
    "NAM",
    "NGA",
    "PAK",
    "PRY",
    "PER",
    "PHL",
    "WSM",
    "LKA",
    "SWZ",
    "TUN",
    "TKM",
    "UKR",
    "UZB",
    "VNM",
    # Upper-middle income (UMI)
    "DZA",
    "ATG",
    "ARG",
    "AZE",
    "BHS",
    "BRB",
    "BWA",
    "BRA",
    "BGR",
    "CHL",
    "COL",
    "CRI",
    "HRV",
    "CUB",
    "DOM",
    "GAB",
    "GRC",
    "HUN",
    "IRN",
    "IRQ",
    "KAZ",
    "LVA",
    "LBN",
    "LBY",
    "MYS",
    "MUS",
    "MEX",
    "MNE",
    "PAN",
    "POL",
    "ROU",
    "RUS",
    "SRB",
    "SYC",
    "ZAF",
    "SUR",
    "MKD",
    "THA",
    "TUR",
    "URY",
    "VEN",
]

eora.Y.loc[south_codes, eu_countries].sum(axis=1)

final_demand_for_electronics_eu = (
    eora.Y.xs(key="Electrical and Machinery", level="sector")[eu_countries]
    .sum(axis=1)
    .drop(labels=eu_countries)
)
L_non_eu = (
    eora.L.drop(index=eu_countries, level=0)
    .drop(columns=eu_countries, level=0)
    .xs(key="Electrical and Machinery", level="sector", axis=1)
)

L_non_eu.dot(final_demand_for_electronics_eu).nlargest(30)

final_demand_eu = eora.Y[eu_countries].sum(axis=1)

final_demand_eu.loc[
    :,
    pd.IndexSlice[
        "Electrical and Machinery",
        "Transport Equipment",
        "Electricity, Gas and Water",
        "Retail Trade",
        "Transport",
    ],
    :,
].groupby(level="sector").sum()
