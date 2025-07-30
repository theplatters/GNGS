import os
import pandas as pd


def read_t(path):
    datapath = os.path.join(path, "T.csv")
    row_indices_path = os.path.join(path, "index_t.csv")
    col_indices_path = os.path.join(path, "index_t.csv")
    t = pd.read_csv(datapath, header=None)
    row_index_raw = pd.read_csv(
        row_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
    )
    col_index_raw = pd.read_csv(
        col_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
    )

    row_index = pd.MultiIndex.from_frame(
        row_index_raw[["CountryA3", "Entity", "Sector"]]
    )
    col_index = pd.MultiIndex.from_frame(
        col_index_raw[["CountryA3", "Entity", "Sector"]]
    )
    t.index = row_index
    t.columns = col_index
    return t


def read_v(path):
    datapath = os.path.join(path, "V.csv")
    row_indices_path = os.path.join(path, "index_t.csv")
    col_indices_path = os.path.join(path, "index_v.csv")
    t = pd.read_csv(datapath, header=None)
    col_index_raw = pd.read_csv(
        row_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
    )
    row_index_raw = pd.read_csv(
        col_indices_path, delimiter=",", quotechar='"', skipinitialspace=True
    )

    row_index = pd.MultiIndex.from_frame(
        row_index_raw[["CountryA3", "Entity", "Sector"]]
    )
    col_index = pd.MultiIndex.from_frame(
        col_index_raw[["CountryA3", "Entity", "Sector"]]
    )
    t.index = row_index
    t.columns = col_index
    return t


def read_q(path):
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
    row_index = pd.MultiIndex.from_frame(row_index_raw[["IndicatorCode", "LineItems"]])

    q = pd.read_csv(datapath, header=None)
    q.index = row_index
    q.columns = col_index
    return q


class EoraLoader:
    def __init__(self, path):
        self.q = read_q(path)
        self.t = read_t(path)
        self.v = read_v(path)


eora = EoraLoader("data/full_eora")
eora.q
