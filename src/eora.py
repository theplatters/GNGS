import os
import pandas as pd
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
    q: pd.DataFrame
    t: pd.DataFrame
    v: pd.DataFrame
    x: pd.DataFrame
    a: pd.DataFrame
    l: pd.DataFrame

    def __init__(self, path):
        """
        Initializes the EORA.

        Parameters:
            path (str): The relative path where the data is stored.
                        Note that the indices should be left as downloaded, but the data_csvs should be renamed to Y.csv, T.csv, Q.scv, V.csv respectively.
        """
        self.y = self._read_y(path)
        self.q = self._read_q(path)
        self.t = self._read_t(path)
        self.v = self._read_v(path)
        self.x = self.t.sum(axis=0) + self.y.sum(axis=1)
        self.a = self.t.divide(self.x, axis=1)
        self.l = pd.DataFrame(
            inv(np.eye(self.a.shape[0]) - self.a.values),
            index=self.a.index,
            columns=self.a.columns,
        )

    def _read_dataframe(
        self, datapath: str, col_indices_path: str, row_indices_path: str
    ):
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

    def _read_t(self, path):
        datapath = os.path.join(path, "T.csv")
        row_indices_path = os.path.join(path, "index_t.csv")
        col_indices_path = os.path.join(path, "index_t.csv")
        return self._read_dataframe(datapath, col_indices_path, row_indices_path)

    def _read_v(self, path):
        datapath = os.path.join(path, "V.csv")
        row_indices_path = os.path.join(path, "index_v.csv")
        col_indices_path = os.path.join(path, "index_t.csv")
        return self._read_dataframe(datapath, col_indices_path, row_indices_path)

    def _read_y(self, path):
        datapath = os.path.join(path, "Y.csv")
        row_indices_path = os.path.join(path, "index_t.csv")
        col_indices_path = os.path.join(path, "index_y.csv")
        return self._read_dataframe(datapath, col_indices_path, row_indices_path)

    def _read_q(self, path):
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

        col_index = pd.MultiIndex.from_frame(
            col_index_raw[["CountryA3", "Entity", "Sector"]]
        )
        row_index = pd.MultiIndex.from_frame(
            row_index_raw[["IndicatorCode", "LineItems"]]
        )

        q = pd.read_csv(datapath, header=None)
        q.index = row_index
        q.columns = col_index
        return q

    def aggregate(self, sectors: list[tuple], aggregated_sector_name: tuple):
        """Aggregates sectors together

        Args:
            sectors (list[tuplse[str]]): The sectors that are unified
            aggregated_sector_name (tuple[str]): The name of the new sectors

        """
        self.t[aggregated_sector_name] = self.t[sectors].sum(axis=1)
        self.t.loc[aggregated_sector_name] = self.t.loc[sectors].sum(axis=0)
        self.t.drop(sectors, inplace=True)
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

    def dissaggregate(self, sector: tuple, aggregates_into: DisaggregatesInto):
        """dissaggregates a sector into multiple sectors

        Args:
            sector (tuple): The sectors that is deleted             aggregated_sector_name (tuple[str]): The name of the new sectors
            aggregates_into (DisaggregatesInto):

        """
        # delete original sector
        self.t.drop(sector, inplace=True)
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


def test_eora():
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
