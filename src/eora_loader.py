import os
import pandas as pd
import numpy as np
from numpy.linalg import inv


class EoraLoader:
    def __init__(self, path):
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

    def _read_dataframe(self, datapath, col_indices_path, row_indices_path):
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


eora = EoraLoader("data/full_eora")
