"""Compact paired evaluation and required aggregate statistics."""

import hashlib
import json
from pathlib import Path
import statistics


def _percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize_recap(jsonl_path):
    path = Path(jsonl_path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No evaluation records in {path}")
    acceptance = [r["average_accept_length"] for r in records]
    speedup = [r["speedup"] for r in records]
    return {
        "method": "recap", "num_records": len(records),
        "avg_accept_length": statistics.mean(acceptance),
        "median_accept_length": statistics.median(acceptance),
        "max_accept_length": max(acceptance),
        "p90_accept_length": _percentile(acceptance, 0.9),
        "avg_tokens_per_iteration": statistics.mean(r["average_tokens_per_iteration"] for r in records),
        "avg_sample_speedup": statistics.mean(speedup),
        "median_sample_speedup": statistics.median(speedup),
        "max_sample_speedup": max(speedup),
        "sample_speedup_gt_1_ratio": sum(v > 1 for v in speedup) / len(records),
        "accept_le_0_5_ratio": sum(v <= 0.5 for v in acceptance) / len(records),
        "output_token_match_ratio": statistics.mean(r["output_token_match"] for r in records),
        "jsonl_path": str(path.resolve()),
    }


def recap_record(identifier, target, recap):
    def tokens(output):
        return output.sequences[0, output.prompt_length:].tolist()

    def digest(ids):
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    target_ids, recap_ids = tokens(target), tokens(recap)
    iterations = len(recap.acceptance_lengths)
    return {
        "id": identifier, "method": "recap",
        "generated_tokens": recap.generated_tokens,
        "target_generated_tokens": target.generated_tokens,
        "elapsed_seconds": recap.elapsed_seconds,
        "target_elapsed_seconds": target.elapsed_seconds,
        "tokens_per_second": recap.tokens_per_second,
        "target_tokens_per_second": target.tokens_per_second,
        "speedup": recap.tokens_per_second / target.tokens_per_second if target.tokens_per_second else 0.0,
        "iterations": iterations,
        # Accepted draft nodes exclude the already-generated root and correction.
        "average_accept_length": statistics.mean(recap.acceptance_lengths) if iterations else 0.0,
        "average_tokens_per_iteration": (recap.generated_tokens - 1) / iterations if iterations else 0.0,
        "acceptance_lengths": recap.acceptance_lengths,
        "candidate_counts": recap.candidate_counts,
        "output_token_match": target_ids == recap_ids,
        "output_token_hash": digest(recap_ids),
        "target_token_hash": digest(target_ids),
    }
