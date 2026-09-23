"""
difficulty_lookup.py — loads the heuristic-measured difficulty table
(classify_state_difficulty.py's output) and provides a fast
nearest-neighbor difficulty(d0, v0) lookup, cheap enough to call at
every env.reset() during training (unlike re-running the heuristic
live, which would roughly double every episode's cost).
"""
import csv


class DifficultyLookup:
    def __init__(self, csv_path: str):
        self._grid = {}
        d_values, v_values = set(), set()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d0, v0, difficulty = round(float(row["d0"]), 6), round(float(row["v0"]), 6), float(row["difficulty"])
                self._grid[(d0, v0)] = difficulty
                d_values.add(d0)
                v_values.add(v0)
        if not self._grid:
            raise ValueError(f"{csv_path} contained no rows")
        self._d_values = sorted(d_values)
        self._v_values = sorted(v_values)
        self.min_difficulty = min(self._grid.values())
        self.max_difficulty = max(self._grid.values())
        # The single easiest known point -- used as a safe fallback if
        # rejection sampling can't find a match under a very tight threshold.
        self.easiest_point = min(self._grid, key=lambda k: self._grid[k])

    def difficulty(self, d0: float, v0: float) -> float:
        nearest_d = min(self._d_values, key=lambda x: abs(x - d0))
        nearest_v = min(self._v_values, key=lambda x: abs(x - v0))
        return self._grid[(nearest_d, nearest_v)]