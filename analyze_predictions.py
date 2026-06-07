"""
Analyze merged_predictions.json:
  - Runtime comparison between proposed_method and vanilla
  - Cases where their predictions differ from each other
"""

import json
import statistics
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "compare_outputs" / "merged_predictions.json"
OUTPUT_FILE = Path(__file__).parent / "compare_outputs" / "analysis_results.json"


def load_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_entries(data: list[dict]) -> dict:
    proposed_times = []
    vanilla_times = []
    differing_predictions = []

    for entry in data:
        pm = entry["proposed_method"]
        vn = entry["vanilla"]

        proposed_times.append(pm["generation_time_s"])
        vanilla_times.append(vn["generation_time_s"])

        if pm["prediction"] != vn["prediction"]:
            differing_predictions.append(
                {
                    "example_id": entry["example_id"],
                    "question": entry["question"],
                    "ground_truth": entry["ground_truth"],
                    "proposed_method_prediction": pm["prediction"],
                    "vanilla_prediction": vn["prediction"],
                    "proposed_method_time_s": pm["generation_time_s"],
                    "vanilla_time_s": vn["generation_time_s"],
                }
            )

    total = len(data)
    n_diff = len(differing_predictions)

    def safe_speedup(p_mean, v_mean):
        if p_mean == 0:
            return None
        return round(v_mean / p_mean, 4)

    per_sample_speedups = [
        v / p if p > 0 else None
        for p, v in zip(proposed_times, vanilla_times)
    ]
    valid_speedups = [s for s in per_sample_speedups if s is not None]

    runtime_stats = {
        "total_samples": total,
        "proposed_method": {
            "total_time_s": round(sum(proposed_times), 4),
            "mean_time_s": round(statistics.mean(proposed_times), 6),
            "median_time_s": round(statistics.median(proposed_times), 6),
            "stdev_time_s": round(statistics.stdev(proposed_times), 6),
            "min_time_s": round(min(proposed_times), 6),
            "max_time_s": round(max(proposed_times), 6),
        },
        "vanilla": {
            "total_time_s": round(sum(vanilla_times), 4),
            "mean_time_s": round(statistics.mean(vanilla_times), 6),
            "median_time_s": round(statistics.median(vanilla_times), 6),
            "stdev_time_s": round(statistics.stdev(vanilla_times), 6),
            "min_time_s": round(min(vanilla_times), 6),
            "max_time_s": round(max(vanilla_times), 6),
        },
        # speedup > 1 means proposed_method is faster
        "speedup_vanilla_over_proposed": {
            "total": safe_speedup(sum(proposed_times), sum(vanilla_times)),
            "mean_per_sample": round(statistics.mean(valid_speedups), 4),
            "median_per_sample": round(statistics.median(valid_speedups), 4),
        },
        "proposed_faster_count": sum(1 for p, v in zip(proposed_times, vanilla_times) if p < v),
        "vanilla_faster_count": sum(1 for p, v in zip(proposed_times, vanilla_times) if v < p),
        "equal_time_count": sum(1 for p, v in zip(proposed_times, vanilla_times) if p == v),
    }

    prediction_diff_stats = {
        "total_samples": total,
        "different_prediction_count": n_diff,
        "same_prediction_count": total - n_diff,
        "different_prediction_rate": round(n_diff / total, 4) if total else 0,
        "differing_cases": differing_predictions,
    }

    return {
        "runtime_analysis": runtime_stats,
        "prediction_difference_analysis": prediction_diff_stats,
    }


def main():
    data = load_data(INPUT_FILE)
    results = compare_entries(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rt = results["runtime_analysis"]
    pd = results["prediction_difference_analysis"]

    print("=== Runtime Summary ===")
    print(f"  Total samples          : {rt['total_samples']}")
    print(f"  Proposed total time    : {rt['proposed_method']['total_time_s']:.2f}s  "
          f"(mean {rt['proposed_method']['mean_time_s']:.4f}s)")
    print(f"  Vanilla   total time   : {rt['vanilla']['total_time_s']:.2f}s  "
          f"(mean {rt['vanilla']['mean_time_s']:.4f}s)")
    speedup = rt["speedup_vanilla_over_proposed"]["total"]
    direction = "proposed faster" if speedup and speedup > 1 else "vanilla faster"
    print(f"  Overall speedup ratio  : {speedup:.4f}x  ({direction})")
    print(f"  Proposed faster (per sample): {rt['proposed_faster_count']} / {rt['total_samples']}")
    print(f"  Vanilla   faster (per sample): {rt['vanilla_faster_count']} / {rt['total_samples']}")

    print("\n=== Prediction Difference Summary ===")
    print(f"  Different predictions  : {pd['different_prediction_count']} / {pd['total_samples']} "
          f"({pd['different_prediction_rate']*100:.1f}%)")
    print(f"  Same predictions       : {pd['same_prediction_count']}")

    print(f"\nFull results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
