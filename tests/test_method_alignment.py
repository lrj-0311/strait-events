"""Small data-free checks for the released alignment rules."""
import unittest

import pandas as pd

from scripts.prepare_context import FEATURES, prepare_context, prepare_groups
from scripts.analyze_associations import add_future_labels, full_week_bounds


class AlignmentTests(unittest.TestCase):
    def test_unobserved_calendar_week_is_not_zero(self):
        dates = pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-27"])
        frame = pd.DataFrame({"count": [1, 2, 3]}, index=dates)
        shifted = add_future_labels(frame, ["count"], lags=(1,))
        self.assertEqual(shifted.iloc[0]["count_future_1w"], 2)
        self.assertTrue(pd.isna(shifted.iloc[1]["count_future_1w"]))
        self.assertEqual(full_week_bounds("2020-01-08", "2020-02-19")[0], pd.Timestamp("2020-01-13"))

    def test_context_uses_previous_week(self):
        dates = ["2020-01-06", "2020-01-13", "2020-01-20"]
        mechanism = pd.DataFrame({"timestamp_start": dates, **{
            feature: [1.0, 2.0, 3.0] for feature in FEATURES
        }})
        slices = pd.DataFrame({
            "timestamp_id": [0, 1], "timestamp_start": dates[1:], "split": ["train", "test"]
        })
        context = prepare_context(mechanism, slices)
        self.assertEqual(context["avg_tone_lag_1"].tolist(), [1.0, 2.0])
        self.assertEqual(len([name for name in context if name.endswith("_lag_1")]), 9)
        self.assertFalse(any(name.startswith("z::") for name in context))

    def test_core_groups_are_exclusive(self):
        entities = pd.DataFrame({"id": [0, 1, 2], "name": [
            "ACTOR::China", "ACTOR::United States", "ACTOR::China-Japan"
        ]})
        groups = prepare_groups(entities)
        self.assertEqual(groups["is_core"].tolist(), [1, 0, 1])
        self.assertEqual(groups["is_external"].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
