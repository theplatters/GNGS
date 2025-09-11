import src.eora as eo
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

eora = eo.Eora("data/full_eora")

eora.t.loc["FRA", "Commodities", "Motor vehicles, trailers and semi-trailers"].T.nlargest(50, columns=("FRA", "Commodities", "Motor vehicles, trailers and semi-trailers"))

eora.t["FRA", "Commodities", "Motor vehicles, trailers and semi-trailers"].nlargest(50, columns=("FRA", "Commodities", "Motor vehicles, trailers and semi-trailers"))
eora.t.loc["DEU", "Commodities", "Passenger cars and parts"].T.nlargest(50, columns=("DEU", "Commodities", "Passenger cars and parts"))
eora.t["DEU", "Commodities", "Passenger cars and parts"].nlargest(50, columns=("DEU", "Commodities", "Passenger cars and parts"))

eora.t.loc["CHN", "Commodities", "Motor vehicles"].T.nlargest(50, columns=("CHN", "Commodities", "Motor vehicles"))
eora.t["CHN", "Commodities", "Motor vehicles"].nlargest(50, columns=("CHN", "Commodities", "Motor vehicles"))


eora.t.loc["JPN", "Commodities", "Passenger motor cars"].T.nlargest(50, columns=("JPN", "Commodities", "Passenger motor cars"))
eora.t["JPN", "Commodities", "Passenger motor cars"].nlargest(50, columns=("JPN", "Commodities", "Passenger motor cars"))

eora.aggregate(
    [("JPN","Commodities", "Passenger motor cars"),
     ("CHN", "Commodities", "Motor vehicles"),
     ("ESP", "Commodities", "Motor vehicles"),
     ("DEU", "Commodities", "Passenger cars and parts"),
     ("AUT","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("BEL","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("CZE","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("EST","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("FIN","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("FRA","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("GRC","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("HUN","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("IRL","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("LTU","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("LVA","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("MLT","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("NLD","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("POL","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("PRT","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("ROU","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("SVK","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("SVN","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("SWE","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("ITA","Commodities","Motor vehicles, trailers and semi-trailers"),
     ("AUT","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("BEL","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("EST","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("CZE","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("FIN","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("FRA","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("GRC","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("HUN","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("IRL","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("LTU","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("LVA","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("MLT","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("NLD","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("POL","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("PRT","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("ROU","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("SVK","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("SVN","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("SWE","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("ITA","Industries","Manufacture of motor vehicles, trailers and semi-trailers"),
     ("KOR","Commodities", "Motor vehicles and parts")
     ],
    ("CAR","Commodities","Cars")
)

eora.t["CAR","Commodities", "Cars"].nlargest(30, columns=("CAR","Commodities", "Cars"))
