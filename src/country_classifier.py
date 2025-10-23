import json
import pandas as pd

countries = pd.read_excel("data/CLASS_2025_10_07.xlsx")
countries_grouped_by_income = countries.groupby("Income group")
income_levels = countries["Income group"].unique()
for income_level in income_levels:
    data = countries_grouped_by_income.get_group(income_level)["Code"].to_list()
    with open("data/" + income_level.replace(" ", "_") + ".json", "w") as f:
        json.dump(data, f)
