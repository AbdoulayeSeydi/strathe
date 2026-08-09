#!/usr/bin/env python3
"""
Directly tests the suspected mechanism (repeated checkpoint file I/O
triggering occasional Windows-level slowdowns - Defender re-scanning on
write-close, Search Indexer, etc.) WITHOUT running any real physics or
calibration - just calls the exact same write pattern
(make_checkpoint_callback uses: open+yaml.dump to checkpoint.yaml, open
+append to timing_log.csv) hundreds of times in a tight loop, with fake
data, timing every single write.

If this reproduces occasional multi-second+ outliers on its own, that's
real, direct, cheap evidence for the checkpoint-I/O hypothesis - achieved
in minutes, not the hours a full A/B calibration test would cost.

Usage:
    python check_checkpoint_io_directly.py --n-writes 500 --output-dir ../io_test_output
"""
import argparse
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-writes", type=int, default=500)
    parser.add_argument("--output-dir", default="../io_test_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.yaml"
    timing_log_path = output_dir / "timing_log.csv"

    # Fake data matching the REAL checkpoint's actual structure/size -
    # 22 real-looking float parameters, so the write size matches reality
    fake_params = {f"param_{i}": float(i) * 1.2345 for i in range(22)}

    print(f"Writing {args.n_writes} times to {output_dir}/, timing EVERY write, "
          f"no physics/calibration involved - just the exact real I/O pattern.\n")

    write_times = []
    for i in range(args.n_writes):
        t0 = time.monotonic()

        # Exact same pattern as make_checkpoint_callback: full yaml.dump
        # to checkpoint.yaml (overwrite), plus an append to timing_log.csv
        checkpoint = {
            "call_count": i,
            "current_cost": float(i) * 0.01,
            "current_parameters": fake_params,
            "best_cost_so_far": float(i) * 0.01,
            "best_parameters_so_far": fake_params,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            yaml.dump(checkpoint, f, sort_keys=False)

        with open(timing_log_path, "a", encoding="utf-8") as f:
            if i == 0:
                f.write("call_count,unix_timestamp,cost\n")
            f.write(f"{i},{time.time()},{float(i) * 0.01}\n")

        elapsed = time.monotonic() - t0
        write_times.append(elapsed)
        if elapsed > 1.0:
            print(f"  OUTLIER at write {i}: {elapsed:.2f}s (normal is milliseconds)")

    import statistics
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {args.n_writes} writes")
    print(f"  median: {statistics.median(write_times) * 1000:.2f}ms")
    print(f"  mean:   {statistics.mean(write_times) * 1000:.2f}ms")
    print(f"  max:    {max(write_times):.3f}s")
    print(f"  min:    {min(write_times) * 1000:.2f}ms")
    n_outliers = sum(1 for t in write_times if t > 0.5)
    print(f"  writes taking >0.5s: {n_outliers} / {args.n_writes}")
    print(f"{'=' * 70}")
    if max(write_times) > 1.0:
        print("REAL OUTLIER(S) FOUND - this directly supports the checkpoint-I/O hypothesis.")
        print("A single file write taking over a second (should take milliseconds) with")
        print("nothing else running is strong, direct evidence something (Defender, Search")
        print("Indexer, etc.) is occasionally interfering with writes to this location.")
    else:
        print("No real outliers in this run - doesn't rule out the hypothesis (the real")
        print("anomaly is sporadic), but doesn't support it either from this sample alone.")


if __name__ == "__main__":
    main()