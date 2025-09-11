import os
import pandas as pd
from src.extension import Extension
import numpy as np
from numpy.linalg import inv
from typing import Iterable, Any
from dataclasses import dataclass


@dataclass
class SectorData:
    t_rows: pd.Series
    t_columns: pd.Series
    x: pd.Series
    y: pd.Series
    v: pd.Series
    q: pd.Series


DisaggregatesInto = Iterable[tuple[Any, SectorData]]


class Eora:
    """Loads in the full eora

    Attributes:
        y (pd.DataFrame): The final demand vector
        q (pd.DataFrame): The sattelite accounts
        t (pd.DataFrame): The transaction matrix
        v (pd.DataFrame): The value added matrix
        a (pd.DataFrame): The technical coefficients matrix
        l (pd.DataFrame): The leontief inverse
    """

    y: pd.DataFrame
    q: Extension
    t: pd.DataFrame
    v: pd.DataFrame
    x: pd.DataFrame
    a: pd.DataFrame
    l: pd.DataFrame

    def __init__(self, path) -> None:
        """
        Initializes the EORA.

        Parameters:
            path (str): The relative path where the data is stored.
                        Note that the indices should be left as downloaded, but the data_csvs should be renamed to Y.csv, T.csv, Q.scv, V.csv respectively.
        """
        self.y = self._read_y(path)
        self.t = self._read_t(path)
        self.v = self._read_v(path)
        self.x = self.t.sum(axis=0) + self.y.sum(axis=1)
        q, q_y = self._read_q(path)
        self.q = Extension(q, q_y, self.x)
        self.a = self.t.divide(self.x, axis=1)
        self.l = pd.DataFrame(
            inv(np.eye(self.a.shape[0]) - self.a.values),
            index=self.a.index,
            columns=self.a.columns,
        )

    def _read_dataframe(
        self, datapath: str, col_indices_path: str, row_indices_path: str
    ) -> pd.DataFrame:
        t = pd.read_csv(datapath, header=None)
        col_index_raw = pd.read_csv(
            col_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
        )

        row_index_raw = pd.read_csv(
            row_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
        )

        col_index = pd.MultiIndex.from_frame(
            col_index_raw[["CountryA3", "Entity", "Sector"]]
        )

        row_index = pd.MultiIndex.from_frame(
            row_index_raw[["CountryA3", "Entity", "Sector"]]
        )
        t.columns = col_index
        t.index = row_index
        return t

    def _read_t(self, path: str) -> pd.DataFrame:
        datapath = os.path.join(path, "T.csv")
        row_indices_path = os.path.join(path, "index_t.csv")
        col_indices_path = os.path.join(path, "index_t.csv")
        return self._read_dataframe(datapath, col_indices_path, row_indices_path)

    def _read_v(self, path: str) -> pd.DataFrame:
        datapath = os.path.join(path, "V.csv")
        row_indices_path = os.path.join(path, "index_v.csv")
        col_indices_path = os.path.join(path, "index_t.csv")
        return self._read_dataframe(datapath, col_indices_path, row_indices_path)

    def _read_y(self, path: str) -> pd.DataFrame:
        datapath = os.path.join(path, "Y.csv")
        row_indices_path = os.path.join(path, "index_t.csv")
        col_indices_path = os.path.join(path, "index_y.csv")
        return self._read_dataframe(datapath, col_indices_path, row_indices_path)

    def _read_q(self, path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        datapath = os.path.join(path, "Q.csv")
        row_indices_path = os.path.join(path, "index_q.csv")
        col_indices_path_t = os.path.join(path, "index_t.csv")
        col_indices_path_y = os.path.join(path, "index_y.csv")
        col_index_raw_t = pd.read_csv(
            col_indices_path_t, delimiter=",", quotechar='"', skipinitialspace=True
        )
        col_index_raw_y = pd.read_csv(
            col_indices_path_y, delimiter=",", quotechar='"', skipinitialspace=True
        )

        col_index_raw = pd.concat([col_index_raw_t, col_index_raw_y])
        row_index_raw = pd.read_csv(
            row_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
        )

        col_index_y = pd.MultiIndex.from_frame(
            col_index_raw_y[["CountryA3", "Entity", "Sector"]]
        )

        col_index = pd.MultiIndex.from_frame(
            col_index_raw[["CountryA3", "Entity", "Sector"]]
        )
        row_index = pd.MultiIndex.from_frame(
            row_index_raw[["IndicatorCode", "LineItems"]]
        )

        q: pd.DataFrame = pd.read_csv(datapath, header=None)
        q.index = row_index
        q.columns = col_index
        q_y = q.loc[:, col_index_y]
        q_rest = q.drop(col_index_y.to_list())
        return q_rest, q_y

    def aggregate(self, sectors: list[tuple], aggregated_sector_name: tuple) -> None:
        """Aggregates sectors together

        Args:
            sectors (list[tuplse[str]]): The sectors that are unified
            aggregated_sector_name (tuple[str]): The name of the new sectors

        """
        self.t[aggregated_sector_name] = self.t[sectors].sum(axis=1)
        self.t.loc[aggregated_sector_name] = self.t.loc[sectors].sum(axis=0)
        self.t.drop(sectors, axis=0, inplace=True)
        self.t.drop(sectors, axis=1, inplace=True)

        self.x[aggregated_sector_name] = self.x[sectors].sum()
        self.x.drop(sectors, inplace=True)

        self.y.loc[aggregated_sector_name] = self.y.loc[sectors].sum(axis=1)
        self.y.drop(sectors, inplace=True)
        self.q[aggregated_sector_name] = self.q[sectors].sum(axis=1)
        self.q.drop(sectors, axis=1, inplace=True)

        self.a = self.t.divide(self.x, axis=1)
        self.l = pd.DataFrame(
            inv(np.eye(self.a.shape[0]) - self.a.values),
            index=self.a.index,
            columns=self.a.columns,
        )

    def dissaggregate(self, sector: tuple, aggregates_into: DisaggregatesInto) -> None:
        """dissaggregates a sector into multiple sectors

        Args:
            sector (tuple): The sectors that is deleted
            aggregates_into (DisaggregatesInto):

        """
        # delete original sector
        self.t.drop(sector, inplace=True, axis=0)
        self.t.drop(sector, inplace=True, axis=1)
        self.x.drop(sector, inplace=True)
        self.y.drop(sector, inplace=True)
        self.q.drop(sector, axis=1, inplace=True)
        _, data = next(iter(aggregates_into))
        first_index = data.t_columns.index

        if not all(x.t_columns.index.equals(first_index) for _, x in aggregates_into):
            raise ValueError("Not all indices are equal")

        self.t.reindex(index=data.t_columns.index, columns=data.t_rows.index)
        for (
            sector_name,
            data,
        ) in aggregates_into:  # or however you iterate over DisaggregatesInto
            self.t[sector_name] = data.t_columns
            self.t.loc[sector_name] = data.t_rows

            self.x[sector_name] = data.x

            self.y[sector_name] = data.y

            self.v[sector_name] = data.v

            self.q[sector_name] = data.q

        # Recalculate derived matrices
        self.a = self.t.divide(self.x, axis=1)
        self.l = pd.DataFrame(
            inv(np.eye(self.a.shape[0]) - self.a.values),
            index=self.a.index,
            columns=self.a.columns,
        )


def test_eora() -> Eora:
    """Generate a toy Eora instance with 3 countries × 3 sectors."""

    countries = ["USA", "CHN", "DEU"]
    sectors = [f"S{i}" for i in range(1, 4)]
    entity = "Industry"  # simple placeholder entity

    # Build MultiIndex for 9 sectors
    sector_tuples = [(c, entity, s) for c in countries for s in sectors]
    sector_index = pd.MultiIndex.from_tuples(
        sector_tuples, names=["CountryA3", "Entity", "Sector"]
    )

    # Transaction matrix (9x9)
    t = pd.DataFrame(
        np.random.randint(10, 100, size=(9, 9)),
        index=sector_index,
        columns=sector_index,
    )

    # Final demand (9x3)
    y_columns = pd.MultiIndex.from_tuples(
        [(c, "FinalDemand", f"FD{i}") for c in countries for i in range(1, 2)],
        names=["CountryA3", "Entity", "Sector"],
    )
    y = pd.DataFrame(
        np.random.randint(5, 50, size=(9, 3)),
        index=sector_index,
        columns=y_columns,
    )

    # Value added (9x2)
    v_columns = ["VA1", "VA2"]
    v = pd.DataFrame(
        np.random.randint(5, 50, size=(2, 9)),
        index=v_columns,
        columns=sector_index,
    )

    # Satellite accounts (4x9) → emissions
    q_index = pd.MultiIndex.from_tuples(
        [(f"EM{i}", f"Type{i}") for i in range(1, 5)],
        names=["IndicatorCode", "LineItems"],
    )
    q = pd.DataFrame(
        np.random.randint(1, 20, size=(4, 9)),
        index=q_index,
        columns=sector_index,
    )

    # Construct dummy Eora
    eora = Eora.__new__(Eora)  # bypass __init__
    eora.t = t
    eora.y = y
    eora.v = v
    eora.q = q
    eora.x = eora.t.sum(axis=0) + eora.y.sum(axis=1)
    eora.a = eora.t.divide(eora.x, axis=1)
    eora.l = pd.DataFrame(
        np.linalg.inv(np.eye(eora.a.shape[0]) - eora.a.values),
        index=eora.a.index,
        columns=eora.a.columns,
    )
    return eora

import re


#
ICE_PATTERNS = [
    r"\b(coke|refined petroleum|petroleum products|nuclear fuels)\b",
    r"\bbasic ferrous metals?\b",
    r"\bfabricated metal products?\b",
    r"\bmining and quarrying\b",
    r"\bmachinery and equipment\b",
]
EV_PATTERNS = [
    r"\belectrical (and )?machinery\b",
    r"\bcommunication (and )?electronic equipment\b",
    r"\boffice machinery and computers\b",
    r"\bmedical, scientific, optical equipment\b",
    r"\bbasic non-?ferrous metals?\b",
    r"\binsulated wire|cables?\b",
]

from typing import List, Dict, Tuple, Optional, Iterable

COUNTRY_TWEAKS: Dict[str, Dict[str, List[str]]] = {
    # "DEU": {"ICE": [r"..."], "EV": [r"..."]},
    # "CHN": {"EV": [r"electronic element and device"]},
}

def _country_patterns(iso: str) -> Tuple[List[str], List[str]]:
    """Return (ICE_patterns, EV_patterns) for a given ISO code, with country tweaks applied."""
    ice = ICE_PATTERNS.copy()
    ev  = EV_PATTERNS.copy()
    tweaks = COUNTRY_TWEAKS.get(iso.upper(), {})
    ice += tweaks.get("ICE", [])
    ev  += tweaks.get("EV", [])
    return ice, ev


def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9&]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    t = _norm_text(text)
    return any(re.search(p, t) for p in patterns)

def _bundle_row_mask(eora, iso: str, patterns: list[str]) -> pd.Series:
    idx = eora.t.index
    countries = idx.get_level_values(0)
    sectors   = idx.get_level_values(2).astype(str)
    return (countries == iso) & sectors.to_series().apply(lambda s: _matches_any(s, patterns)).values


# ---------- bundle row/col masks ----------
def _bundle_row_mask(eora, iso: str, patterns: list[str]) -> pd.Series:
    """Select *supplying* rows: (CountryA3==iso) & Sector matches bundle."""
    idx = eora.t.index  # MultiIndex (CountryA3, Entity, Sector)
    countries = idx.get_level_values(0)
    sectors   = idx.get_level_values(2).astype(str)
    return (countries == iso) & sectors.to_series().apply(lambda s: _matches_any(s, patterns)).values

def _country_col_mask(eora, iso: str) -> pd.Series:
    """Select *buying* columns for a given country code."""
    cols = eora.t.columns
    return cols.get_level_values(0) == iso

def _not_country_col_mask(eora, iso: str) -> pd.Series:
    cols = eora.t.columns
    return cols.get_level_values(0) != iso

def bundle_demand_stats(eora, iso: str, kind: str = "ICE", top_suppliers: int = 15) -> dict:
    """
    Backward linkages: who supplies inputs to the ICE/EV bundle sectors in `iso`.
    Returns:
      - total_inputs: sum of all inputs into the bundle sectors
      - domestic_inputs: inputs sourced domestically
      - import_inputs: inputs sourced from abroad
      - top_supplier_sectors: Series of inputs by (country, entity, sector)
      - top_supplier_countries: Series aggregated by country
    """
    ice_patterns, ev_patterns = _country_patterns(iso)
    patterns = ice_patterns if kind.upper() == "ICE" else ev_patterns

    # --- select bundle columns (buyers) ---
    col_idx = eora.t.columns
    col_countries = col_idx.get_level_values(0)
    col_sectors   = col_idx.get_level_values(2).astype(str)
    col_mask_bundle = (col_countries == iso) & pd.Series(col_sectors).apply(lambda s: _matches_any(s, patterns)).values

    if not col_mask_bundle.any():
        return {"total_inputs": 0, "domestic_inputs": 0, "import_inputs": 0,
                "top_supplier_sectors": pd.Series(dtype=float),
                "top_supplier_countries": pd.Series(dtype=float)}

     = eora.t.loc[:, col_mask_bundle]

    # total inputs
    total_inputs = T_cols_bundle.values.sum()

    # domestic vs foreign split (by supplier country code in row index)
    row_idx = eora.t.index
    row_countries = row_idx.get_level_values(0)
    domestic_inputs = T_cols_bundle[row_countries == iso].values.sum()
    import_inputs   = total_inputs - domestic_inputs

    # top suppliers by (country, entity, sector)
    top_supplier_sectors = T_cols_bundle.sum(axis=1).sort_values(ascending=False).head(top_suppliers)

    # aggregated by supplier country
    suppliers_by_country = pd.Series(T_cols_bundle.sum(axis=1).values,
                                     index=row_countries).groupby(lambda c: c).sum().sort_values(ascending=False)
    top_supplier_countries = suppliers_by_country.head(top_suppliers)

    return {
        "total_inputs": float(total_inputs),
        "domestic_inputs": float(domestic_inputs),
        "import_inputs": float(import_inputs),
        "top_supplier_sectors": top_supplier_sectors,
        "top_supplier_countries": top_supplier_countries,
    }


# ---------- core: supply-side stats for a bundle in one country ----------
def bundle_supply_stats(eora, iso: str, kind: str = "ICE", top_buyers: int = 10) -> dict:
    """
    Compute supply-side stats for ICE/EV bundle in `iso`.
    Returns:
      - x_bundle: total output of bundle sectors in country
      - va_bundle: total value added (sum across VA rows for bundle columns)
      - domestic_sales: T from bundle rows -> domestic columns
      - export_sales:   T from bundle rows -> foreign columns
      - top_foreign_buyers: Series of exports by buyer country
      - A_intensity_by_buyer_country (optional signal): sum of A rows in bundle grouped by buyer country
    """
    ice_patterns, ev_patterns = _country_patterns(iso)
    patterns = ice_patterns if kind.upper() == "ICE" else ev_patterns

    # masks
    row_mask = _bundle_row_mask(eora, iso, patterns)              # supplier rows (in iso, matching bundle)
    dom_cols = _country_col_mask(eora, iso)                        # domestic buyers
    for_cols = _not_country_col_mask(eora, iso)                    # foreign buyers

    # 1) supply capacity
    # x is indexed by columns (sectors): pick supplier columns in iso matching bundle
    col_idx = eora.t.columns
    col_countries = col_idx.get_level_values(0)
    col_sectors   = col_idx.get_level_values(2).astype(str)
    col_mask_bundle = (col_countries == iso) & pd.Series(col_sectors).apply(lambda s: _matches_any(s, patterns)).values
    x_bundle = eora.x[col_mask_bundle].sum()

    # 2) value added of those supplier sectors
    # v has rows = VA items, columns = sectors (same MultiIndex as T columns)
    va_bundle = eora.v.loc[:, col_mask_bundle].sum().sum()

    # 3) sales flows from those supplier rows (T)
    T_rows_bundle = eora.t.loc[row_mask, :]
    domestic_sales = T_rows_bundle.loc[:, dom_cols].values.sum()
    export_sales   = T_rows_bundle.loc[:, for_cols].values.sum()

    # 4) top foreign buyer countries (sum across all buyer sectors, grouped by buyer country)
    buyer_countries = eora.t.columns.get_level_values(0)
    exports_by_country = pd.Series(T_rows_bundle.loc[:, for_cols].sum(axis=0).values,
                                   index=buyer_countries[for_cols]).groupby(lambda c: c).sum().sort_values(ascending=False)
    top_foreign_buyers = exports_by_country.head(top_buyers)

    # 5) optional “presence” signal in A (sum supplier rows of A over bundle; group by buyer country)
    A_rows_bundle = eora.a.loc[row_mask, :]
    A_by_buyer_country = pd.Series(A_rows_bundle.sum(axis=0).values,
                                   index=eora.a.columns.get_level_values(0)).groupby(lambda c: c).sum().sort_values(ascending=False)

    return {
        "x_bundle": float(x_bundle),
        "va_bundle": float(va_bundle),
        "domestic_sales": float(domestic_sales),
        "export_sales": float(export_sales),
        "top_foreign_buyers": top_foreign_buyers,
        "A_intensity_by_buyer_country": A_by_buyer_country,
    }

# ---------- convenience: both bundles + ratios ----------
def bundle_pairs_for_country(eora, iso: str) -> dict:
    ice = bundle_supply_stats(eora, iso, "ICE", top_buyers=10)
    ev  = bundle_supply_stats(eora, iso, "EV",  top_buyers=10)

    def ratio(n, d):
        return float(n/d) if d and np.isfinite(n) and np.isfinite(d) else np.nan

    return {
        "ICE": ice,
        "EV": ev,
        "ratios": {
            "x_EV_to_ICE": ratio(ev["x_bundle"], ice["x_bundle"]),
            "va_EV_to_ICE": ratio(ev["va_bundle"], ice["va_bundle"]),
            "exports_EV_to_ICE": ratio(ev["export_sales"], ice["export_sales"]),
            "domestic_EV_to_ICE": ratio(ev["domestic_sales"], ice["domestic_sales"]),
        }
    }
if __name__ == "__main__":
    data_dir = "data/eora"
    eora = Eora(data_dir)
    print("Loaded Eora from:", data_dir)

    countries = ["DEU","FRA","ITA","ESP","CHN"]
    rows = []
    for iso in countries:
        out = bundle_pairs_for_country(eora, iso)
        ice, ev, r = out["ICE"], out["EV"], out["ratios"]

        print(f"\n=== {iso} ===")
        print(f"ICE  — x:{ice['x_bundle']:.3g}  VA:{ice['va_bundle']:.3g}  domestic:{ice['domestic_sales']:.3g}  exports:{ice['export_sales']:.3g}")
        print(f"EV   — x:{ev['x_bundle']:.3g}  VA:{ev['va_bundle']:.3g}  domestic:{ev['domestic_sales']:.3g}  exports:{ev['export_sales']:.3g}")
        print("Top foreign buyers of ICE bundle:\n", ice["top_foreign_buyers"])
        print("Top foreign buyers of EV  bundle:\n", ev["top_foreign_buyers"])

        rows.append({
            "country": iso,
            "x_ICE": ice["x_bundle"], "x_EV": ev["x_bundle"], "x_EV_to_ICE": r["x_EV_to_ICE"],
            "va_ICE": ice["va_bundle"], "va_EV": ev["va_bundle"], "va_EV_to_ICE": r["va_EV_to_ICE"],
            "domestic_ICE": ice["domestic_sales"], "domestic_EV": ev["domestic_sales"],
            "exports_ICE": ice["export_sales"], "exports_EV": ev["export_sales"],
            "exports_EV_to_ICE": r["exports_EV_to_ICE"], "domestic_EV_to_ICE": r["domestic_EV_to_ICE"],
        })

    # ----- BACKWARD LINKAGES: save a tidy summary for all countries -----
    rows_back = []
    for iso in countries:
        back_ice = bundle_demand_stats(eora, iso, "ICE", top_suppliers=10)
        back_ev = bundle_demand_stats(eora, iso, "EV", top_suppliers=10)


        def ratio(n, d):
            return (n / d) if d else np.nan


        rows_back.append({
            "country": iso,
            "inputs_total_ICE": back_ice["total_inputs"],
            "inputs_domestic_ICE": back_ice["domestic_inputs"],
            "inputs_imports_ICE": back_ice["import_inputs"],
            "inputs_total_EV": back_ev["total_inputs"],
            "inputs_domestic_EV": back_ev["domestic_inputs"],
            "inputs_imports_EV": back_ev["import_inputs"],
            "ratio_total_EV_to_ICE": ratio(back_ev["total_inputs"], back_ice["total_inputs"]),
            "ratio_imports_EV_to_ICE": ratio(back_ev["import_inputs"], back_ice["import_inputs"]),
            "ratio_domestic_EV_to_ICE": ratio(back_ev["domestic_inputs"], back_ice["domestic_inputs"]),
        })

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows_back).to_csv("outputs/bundle_backward_summary.csv", index=False)
    print("Saved: outputs/bundle_backward_summary.csv")
