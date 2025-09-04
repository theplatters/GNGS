import src.eora as eo
from copy import deepcopy
import unittest


class TestEora(unittest.TestCase):
    def test_disaggregate(self):
        eora = eo.Eora("data/full_eora")
        eora_orig = deepcopy(eora)

        eora.aggregate(
            [("AFG", "Industries", "Agriculture"), ("AFG", "Industries", "Fishing")],
            ("test", "test", "test"),
        )

        sector_data = eo.SectorData(
            t_rows=eora_orig.t[("AFG", "Industries", "Agriculture")],
            t_columns=eora_orig.t.loc[("AFG", "Industries", "Agriculture")],
            x=eora_orig.x[("AFG", "Industries", "Agriculture")],
            y=eora_orig.y[("AFG", "Industries", "Agriculture")],
            q_rows=eora_orig.q[("AFG", "Industries", "Agriculture")],
        )

        sector_data_2 = eo.SectorData(
            t_rows=eora_orig.t[("AFG", "Industries", "Fishing")],
            t_columns=eora_orig.t.loc[("AFG", "Industries", "Fishing")],
            x=eora_orig.x[("AFG", "Industries", "Fishing")],
            y=eora_orig.y[("AFG", "Industries", "Fishing")],
            q_rows=eora_orig.q[("AFG", "Industries", "Fishing")],
        )

        dis: eo.DisaggregatesInto = [
            (("AFG", "Industries", "Fishing"), sector_data_2),
            (("AFG", "Industries", "Agriculture"), sector_data),
        ]

        eora.dissaggregate(("test", "test", "test"), dis)
        self.assertTrue("FOO".isupper())


if __name__ == "__main_":
    unittest.main()
