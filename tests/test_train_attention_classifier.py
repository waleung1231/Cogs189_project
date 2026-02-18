import unittest

import pandas as pd

import train_attention_classifier as tac


class TrainClassifierTests(unittest.TestCase):
    def test_get_stratified_n_splits_reduces_for_small_minority_class(self):
        y = [0, 0, 0, 1, 1]
        self.assertEqual(tac._get_stratified_n_splits(y, max_splits=5), 2)

    def test_get_stratified_n_splits_returns_zero_for_single_class(self):
        y = [1, 1, 1, 1]
        self.assertEqual(tac._get_stratified_n_splits(y, max_splits=5), 0)

    def test_assign_labels_marks_misses_and_slow_hits_as_zoned_out(self):
        meta = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "is_target": [1, 1, 1, 0],
                "response": ["hit", "hit", "miss", "false_alarm"],
                "rt": [0.2, 1.2, float("nan"), 0.1],
            }
        )

        labeled = tac.assign_labels(meta)
        # miss must be zoned_out=1
        miss_row = labeled[labeled["response"] == "miss"].iloc[0]
        self.assertEqual(miss_row["zoned_out"], 1)

    def test_validate_metadata_schema_accepts_required_columns(self):
        meta = pd.DataFrame(
            {
                "index": [0],
                "is_target": [1],
                "response": ["hit"],
                "rt": [0.3],
            }
        )
        tac.validate_metadata_schema(meta)


if __name__ == "__main__":
    unittest.main()
