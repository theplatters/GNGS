import pandas as pd


class Extension:
    f: pd.DataFrame
    f_y: pd.DataFrame
    a: pd.DataFrame

    def __init__(self, f: pd.DataFrame, f_y: pd.DataFrame, x: pd.Series) -> None:
        self.f = f
        self.f_y = f_y
        self.a = f.div(x)
