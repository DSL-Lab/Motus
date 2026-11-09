"""Success rate, successful simulated time, and matched successful-pair speedup."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(root):
    groups = defaultdict(list)
    for path in sorted(Path(root).glob("*x/seed*/episodes.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["configuration"].get("original_execution", False):
                continue
            options = record["action_options"]
            key = (
                record["task"],
                record["task_config"],
                options["action_adapter"],
                options["speed_factor"],
            )
            groups[key].append(record)

    def match_key(r):
        config = r["configuration"]
        return (
            r["task"],
            r["task_config"],
            r["scene_seed"],
            r["policy_seed"],
            r["instruction"],
            config.get("ckpt_setting"),
            config.get("wan_path"),
            config.get("vlm_path"),
        )

    baseline = {
        match_key(r): r
        for (_, _, adapter, speed), records in groups.items()
        if adapter == "dense" and speed == 1
        for r in records
        if r["success"]
    }
    rows = []
    for (task, task_config, adapter, speed), records in sorted(groups.items()):
        successful = [r for r in records if r["success"]]
        ratios = []
        for r in successful:
            paired = baseline.get(match_key(r))
            if paired and r["simulated_seconds"] > 0:
                ratios.append(paired["simulated_seconds"] / r["simulated_seconds"])
        rows.append(
            {
                "task": task,
                "task_config": task_config,
                "adapter": adapter,
                "requested_speed_factor": speed,
                "episodes": len(records),
                "successes": len(successful),
                "success_rate": len(successful) / len(records),
                "successful_mean_simulated_seconds": mean(
                    r["simulated_seconds"] for r in successful
                )
                if successful
                else None,
                "successful_mean_wall_seconds": mean(
                    r["wall_seconds"] for r in successful
                )
                if successful
                else None,
                "achieved_mean_paired_speedup": mean(ratios) if ratios else None,
                "matched_successful_pairs": len(ratios),
                "fallback_count": sum(r["fallback_count"] for r in records),
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = summarize(args.results)
    if not rows:
        parser.error("No measured episodes found; no rollout results to summarize")
    print(json.dumps(rows, indent=2))
