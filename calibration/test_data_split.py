"""
Tests for data_split.py. Run with:
    python3 -m unittest test_data_split.py -v
"""
import unittest

from data_split import extract_date, split_by_date, SplitResult


class TestExtractDate(unittest.TestCase):
    def test_standard_filename(self):
        self.assertEqual(extract_date("odom-12-01-2024-run1.csv"), "12-01-2024")

    def test_different_prefix(self):
        # Issue #2: regex shouldn't be locked to "odom-" specifically
        self.assertEqual(extract_date("hover-15-06-2024-run3.csv"), "15-06-2024")

    def test_full_path_not_just_basename(self):
        # Issue #9: matches the filename, not an accidental hit in a
        # directory component
        path = "/mnt/data/archive-01-01-1999/odom-12-01-2024-run1.csv"
        self.assertEqual(extract_date(path), "12-01-2024")

    def test_malformed_filename_raises(self):
        with self.assertRaises(ValueError):
            extract_date("no_date_here.csv")


class TestSplitByDate(unittest.TestCase):
    def test_duplicate_dates_grouped_together(self):
        files = [
            "odom-12-01-2024-run1.csv",
            "odom-12-01-2024-run2.csv",
            "odom-19-02-2024-run1.csv",
        ]
        result = split_by_date(files, heldout_dates=["19-02-2024"])
        self.assertEqual(result.n_train, 2)
        self.assertEqual(result.n_heldout, 1)
        self.assertIn("odom-12-01-2024-run1.csv", result.train_files)
        self.assertIn("odom-12-01-2024-run2.csv", result.train_files)

    def test_single_date_raises(self):
        # Issue #5: can't split with only one distinct group
        files = ["odom-12-01-2024-run1.csv", "odom-12-01-2024-run2.csv"]
        with self.assertRaises(ValueError):
            split_by_date(files)

    def test_explicit_heldout_dates(self):
        files = [
            "odom-12-01-2024-run1.csv",
            "odom-14-02-2024-run1.csv",
            "odom-19-02-2024-run1.csv",
        ]
        result = split_by_date(files, heldout_dates=["19-02-2024"])
        self.assertEqual(result.heldout_groups, ["19-02-2024"])
        self.assertEqual(sorted(result.train_groups), ["12-01-2024", "14-02-2024"])

    def test_unknown_heldout_date_raises(self):
        # Issue #3
        files = ["odom-12-01-2024-run1.csv", "odom-14-02-2024-run1.csv"]
        with self.assertRaises(ValueError):
            split_by_date(files, heldout_dates=["01-01-2035"])

    def test_automatic_heldout_uses_chronological_not_string_order(self):
        # THE core bug fix (Issue #1). If this were string-sorted,
        # "02-12-2024" (Dec 2) would sort AFTER "15-01-2024" (Jan 15),
        # incorrectly making Dec 2 look like the "most recent" date and
        # holding it out instead of the actually-latest one.
        files = [
            "odom-15-01-2024-run1.csv",  # Jan 15 — earliest
            "odom-02-12-2024-run1.csv",  # Dec 2 — actually latest
        ]
        result = split_by_date(files, heldout_fraction=0.5)
        # The chronologically LATEST date (Dec 2) must be the one held out,
        # not "02-12-2024" being treated as early because it starts with "0".
        self.assertEqual(result.heldout_groups, ["02-12-2024"])
        self.assertEqual(result.train_groups, ["15-01-2024"])

    def test_invalid_heldout_fraction_raises(self):
        # Issue #4
        files = ["odom-12-01-2024-run1.csv", "odom-14-02-2024-run1.csv"]
        with self.assertRaises(ValueError):
            split_by_date(files, heldout_fraction=1.2)
        with self.assertRaises(ValueError):
            split_by_date(files, heldout_fraction=0)
        with self.assertRaises(ValueError):
            split_by_date(files, heldout_fraction=-0.5)

    def test_returns_split_result_dataclass(self):
        # Issue #8
        files = ["odom-12-01-2024-run1.csv", "odom-14-02-2024-run1.csv"]
        result = split_by_date(files, heldout_dates=["14-02-2024"])
        self.assertIsInstance(result, SplitResult)


if __name__ == "__main__":
    unittest.main()