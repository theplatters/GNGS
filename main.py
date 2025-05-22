import matplotlib.pyplot as plt
import seaborn as sns
import pymrio
import pandas as pd  # Good practice for type hinting and direct use if needed
from matplotlib.widgets import Slider
import sys  # For flushing output if run in certain environments
import country_converter as coco

years = range(2000, 2018)
exports_into_eu = []
imports_from_eu = []
exports_into_china = []
imports_from_china = []
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

    exports_into_eu.append(
        eora.Z.loc[
            ("LE", "Mining and Quarrying"),
            ("EU", slice(None)),
        ].sum()
    )
    imports_from_eu.append(
        eora.Z.loc[
            ("EU", "Mining and Quarrying"),
            ("LE", slice(None)),
        ].sum()
    )
    exports_into_china.append(
        eora.Z.loc[
            ("LE", "Mining and Quarrying"),
            ("China", slice(None)),
        ].sum()
    )
    imports_from_china.append(
        eora.Z.loc[
            ("China", "Mining and Quarrying"),
            ("LE", slice(None)),
        ].sum()
    )


plt.plot(years, exports_into_eu, label="Exports into EU")
plt.plot(years, imports_from_eu, label="Imports from EU")

plt.plot(years, exports_into_china, label="Exports into Chine")
plt.plot(years, imports_from_china, label="Imports from China")
plt.xlabel("Year")
plt.ylabel("Trade Value")
plt.title("Trade with EU (2000–2017)")
plt.legend()
plt.grid(True)
plt.savefig("import-export-lithium-exporting-countries.png")


eora = pymrio.parse_eora26(year=year, path="data/" + str(year)).calc_all()

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

mining_data = (
    eora.Z[eu_countries]
    .T.groupby(level=1)
    .sum()
    .T.xs("Mining and Quarrying", level="sector")
)
top_mining = mining_data.sum(axis=1).sort_values(ascending=False).head(10)

# Calculate row and column sums
row_sums = mining_data.sum(axis=1)
col_sums = mining_data.sum(axis=0)

# Set significance thresholds (adjust these as needed)
row_threshold = row_sums.quantile(0.75)  # Top 25% of countries
col_threshold = col_sums.quantile(0.75)  # Top 25% of sectors

# Filter the data
significant_data = mining_data.loc[row_sums >= row_threshold, col_sums >= col_threshold]

# Create the filtered heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    significant_data,
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

top_countries = mining_data.sum(axis=1).sort_values(ascending=False).head(15).index

plt.figure(figsize=(14, 7))
mining_data.loc[top_countries].plot(
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

# --- Global Store for Pre-calculated Data ---
precalculated_data = {}
MIN_YEAR = 2000
MAX_YEAR = 2016


# --- Data Pre-calculation Function ---
def precalculate_all_data():
    """
    Loads and processes EORA data for all years and stores the
    relevant 'mining_data' DataFrame for each year.
    """
    print("Starting data pre-calculation for years 2000-2016...")
    sys.stdout.flush()
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        print(f"  Processing year: {year}...")
        sys.stdout.flush()
        try:
            print(f"    Loading EORA data for {year} from 'data/{year}'...")
            sys.stdout.flush()
            eora = pymrio.parse_eora26(year=year, path="data/" + str(year))

            print(f"    Calculating all impacts for {year}...")
            sys.stdout.flush()
            eora.calc_all()

            print(f"    Extracting 'Mining and Quarrying' data for {year}...")
            sys.stdout.flush()
            # Z matrix structure: rows are (region_origin, sector_origin), columns are (region_dest, sector_dest)
            mining_data_for_year = (
                eora.Z[eu_countries]
                .T.groupby(level=1)  # Group by destination sector
                .sum()
                .T.xs(
                    "Mining and Quarrying", level="sector"
                )  # Filter for Mining and Quarrying origin sector
            )
            precalculated_data[year] = mining_data_for_year
            print(f"    Data for {year} processed and stored.")
            sys.stdout.flush()

        except FileNotFoundError:
            error_msg = f"Data files not found for year {year} in 'data/{year}/'."
            print(f"    ERROR: {error_msg}")
            sys.stdout.flush()
            precalculated_data[year] = None  # Mark as unavailable
        except KeyError as e:
            error_msg = f"KeyError during data extraction for {year} (e.g., 'Mining and Quarrying' or 'sector' level not found): {e}"
            print(f"    ERROR: {error_msg}")
            sys.stdout.flush()
            precalculated_data[year] = None  # Mark as unavailable
        except Exception as e:
            error_msg = f"An error occurred while processing data for {year}:\n{e}"
            print(f"    ERROR: {error_msg}")
            sys.stdout.flush()
            precalculated_data[year] = None  # Mark as unavailable
    print("Data pre-calculation finished.\n")
    sys.stdout.flush()


# --- Plotting Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95, hspace=0.4)

# --- Slider Setup ---
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor="lightgoldenrodyellow")
year_slider = Slider(
    ax=ax_slider,
    label="Year",
    valmin=MIN_YEAR,
    valmax=MAX_YEAR,
    valinit=MIN_YEAR,
    valstep=1,
    valfmt="%0.0f",
)


# --- Main Update Function ---
def update_plots(year_val):
    year = int(year_slider.val)

    ax1.clear()
    ax2.clear()

    print(f"Updating plots for year: {year}...")
    sys.stdout.flush()

    mining_data = precalculated_data.get(year)

    if mining_data is None:
        error_msg = f"Pre-calculated data for year {year} is not available or failed during loading."
        print(f"  INFO: {error_msg}")
        sys.stdout.flush()
        ax1.text(0.5, 0.5, error_msg, ha="center", va="center", color="red", wrap=True)
        ax2.text(0.5, 0.5, "Data unavailable.", ha="center", va="center", color="red")
        fig.canvas.draw_idle()
        return

    if mining_data.empty:
        msg = f"No 'Mining and Quarrying' data found after extraction for {year} (pre-calculated)."
        print(f"  INFO: {msg}")
        sys.stdout.flush()
        ax1.text(0.5, 0.5, msg, ha="center", va="center")
        ax2.text(0.5, 0.5, msg, ha="center", va="center")
        fig.canvas.draw_idle()
        return

    print(f"  Using pre-calculated 'Mining and Quarrying' data for {year}.")
    sys.stdout.flush()

    # --- Plot 1: Heatmap of Significant Contributors ---
    print(f"  Generating heatmap for {year}...")
    sys.stdout.flush()
    row_sums = mining_data.sum(axis=1)
    col_sums = mining_data.sum(axis=0)

    if row_sums.empty or col_sums.empty:
        msg = f"Not enough data for heatmap thresholds for {year} (sums are empty)."
        print(f"  INFO: {msg}")
        sys.stdout.flush()
        ax1.text(0.5, 0.5, msg, ha="center", va="center")
    else:
        row_threshold = row_sums.quantile(0.75)
        col_threshold = col_sums.quantile(0.75)

        significant_rows = row_sums[row_sums >= row_threshold].index
        significant_cols = col_sums[col_sums >= col_threshold].index

        if not significant_rows.empty and not significant_cols.empty:
            valid_significant_cols = [
                col for col in significant_cols if col in mining_data.columns
            ]

            if not valid_significant_cols:
                significant_data = pd.DataFrame()
            else:
                significant_data = mining_data.loc[
                    significant_rows, valid_significant_cols
                ]

            if not significant_data.empty:
                sns.heatmap(
                    significant_data,
                    ax=ax1,
                    cmap="rocket_r",
                    annot=True,
                    fmt=".0f",
                    linewidths=0.5,
                    cbar_kws={"label": "Economic Value (e.g., Million USD)"},
                    annot_kws={"size": 8},
                )
                ax1.set_title(
                    f"Significant Mining Sector Contributions to EU Industries - {year}",
                    pad=20,
                )
                ax1.set_xlabel(
                    "Receiving EU Industries (Top 25% by total value received)"
                )
                ax1.set_ylabel(
                    "Supplying Countries (Top 25% by total value supplied from Mining)"
                )
                ax1.tick_params(axis="x", labelrotation=45, labelsize=8, ha="right")
                ax1.tick_params(axis="y", labelrotation=0, labelsize=8)
            else:
                msg = f"No data meets significance thresholds for heatmap in {year}."
                print(f"  INFO: {msg}")
                sys.stdout.flush()
                ax1.text(0.5, 0.5, msg, ha="center", va="center")
        else:
            msg = f"No rows/columns meet significance thresholds for heatmap in {year}."
            print(f"  INFO: {msg}")
            sys.stdout.flush()
            ax1.text(0.5, 0.5, msg, ha="center", va="center")
    print(f"  Heatmap generated for {year}.")
    sys.stdout.flush()

    # --- Plot 2: Stacked Bar Chart of Top Countries ---
    print(f"  Generating stacked bar chart for {year}...")
    sys.stdout.flush()
    if not mining_data.empty:  # This check might be redundant if already handled above
        top_countries_series = mining_data.sum(axis=1).sort_values(ascending=False)

        if not top_countries_series.empty:
            num_top_countries = min(10, len(top_countries_series))
            top_countries_index = top_countries_series.head(num_top_countries).index

            if not top_countries_index.empty:
                mining_data_top_countries = mining_data.loc[top_countries_index]

                if not mining_data_top_countries.columns.empty:
                    mining_data_top_countries.plot(
                        kind="bar",
                        stacked=True,
                        colormap="viridis",
                        edgecolor="black",
                        linewidth=0.5,
                        ax=ax2,
                    )
                    ax2.set_title(
                        f"Breakdown of Mining & Quarrying Output (Top {num_top_countries} Supplying Countries) - {year}"
                    )
                    ax2.set_ylabel("Total Economic Value (e.g., Million USD)")
                    ax2.set_xlabel("Country")
                    ax2.legend(
                        title="Destination EU Sectors",
                        bbox_to_anchor=(1.02, 1),
                        loc="upper left",
                        fontsize="small",
                    )
                    ax2.grid(axis="y", linestyle="--", alpha=0.7)
                    ax2.tick_params(axis="x", labelrotation=45, labelsize=9, ha="right")
                    ax2.tick_params(axis="y", labelsize=9)
                else:
                    msg = f"No destination sectors to plot for top countries in {year}."
                    print(f"  INFO: {msg}")
                    sys.stdout.flush()
                    ax2.text(0.5, 0.5, msg, ha="center", va="center")
            else:
                msg = f"No top countries to display for bar chart in {year} (after filtering)."
                print(f"  INFO: {msg}")
                sys.stdout.flush()
                ax2.text(0.5, 0.5, msg, ha="center", va="center")
        else:
            msg = f"No country data available for bar chart in {year}."
            print(f"  INFO: {msg}")
            sys.stdout.flush()
            ax2.text(0.5, 0.5, msg, ha="center", va="center")

    print(f"  Stacked bar chart generated for {year}.")
    sys.stdout.flush()

    fig.canvas.draw_idle()
    print(f"Plots updated for year {year}.\n")
    sys.stdout.flush()


# --- Main Execution ---
precalculate_all_data()

# Connect Slider to Update Function
year_slider.on_changed(update_plots)

# Initial Plot
update_plots(year_slider.valinit)

# Display the plot
plt.show()
