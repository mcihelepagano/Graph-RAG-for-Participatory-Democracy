"""
select_policies_stratified.py
=============================
Selects a stratified evaluation sample of 30 policies from the full
71-policy graph, balanced across argument-count DENSITY.

METHOD
------
  1. Count arguments per policy
  2. Sort by count, split into 3 equal terciles (sparse / medium / dense)
  3. Randomly pick N from each tercile (fixed seed for reproducibility)

This replaces the original biased selection ("the 30 biggest policies")
with an unbiased sample that spans the full density range. That lets
you report how performance varies WITH density — turning a hidden bias
into an explicit research finding.

OUTPUT
------
  data/selected_policies.json   — the sample with per-policy metadata
  Prints a drop-in POLICIES = [...] list for policies.py

Usage:
  python select_policies_stratified.py
  python select_policies_stratified.py --per-tercile 12 --seed 99
"""

import os
import json
import random
import argparse


def load_export(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def arg_counts(export):
    """Return {policy_name: total_argument_count}."""
    arguments = export.get("arguments", {})
    counts = {}
    for pname, data in export.get("policies", {}).items():
        ids = data.get("pros", []) + data.get("cons", [])
        n = sum(1 for aid in ids if aid in arguments)
        counts[pname] = n
    return counts


def tercile_bins(counts):
    """Split policies into 3 density bins by argument count.
    Returns (sparse, medium, dense) as lists of policy names."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    n = len(ordered)
    third = n // 3
    sparse = [p for p, _ in ordered[:third]]
    medium = [p for p, _ in ordered[third:2 * third]]
    dense  = [p for p, _ in ordered[2 * third:]]
    return sparse, medium, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="data/neo4j_export_71full.json",
                    help="path to the full 71-policy export")
    ap.add_argument("--per-tercile", type=int, default=10,
                    help="policies to select per density tercile (default 10)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducibility")
    ap.add_argument("--out", default="data/selected_policies.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    k = args.per_tercile

    print("=" * 70)
    print("STRATIFIED POLICY SELECTION — terciles by argument count")
    print("=" * 70)

    export = load_export(args.export)
    counts = arg_counts(export)
    print(f"Export: {len(counts)} policies total")

    sparse, medium, dense = tercile_bins(counts)

    def count_range(tier):
        if not tier:
            return "(empty)"
        cs = [counts[p] for p in tier]
        return f"{min(cs)}–{max(cs)} args"

    print(f"\nTercile sizes:")
    print(f"  sparse : {len(sparse)} policies, {count_range(sparse)}")
    print(f"  medium : {len(medium)} policies, {count_range(medium)}")
    print(f"  dense  : {len(dense)} policies, {count_range(dense)}")

    # Sample from each tercile
    sel_sparse = rng.sample(sparse, min(k, len(sparse)))
    sel_medium = rng.sample(medium, min(k, len(medium)))
    sel_dense  = rng.sample(dense,  min(k, len(dense)))
    selected = sel_sparse + sel_medium + sel_dense

    print(f"\n" + "-" * 70)
    print("SELECTED POLICIES")
    print("-" * 70)
    for label, tier in [("SPARSE", sel_sparse),
                        ("MEDIUM", sel_medium),
                        ("DENSE",  sel_dense)]:
        print(f"\n{label} ({len(tier)} policies):")
        for p in sorted(tier, key=lambda x: counts[x]):
            print(f"    {counts[p]:>4} args  {p}")

    print(f"\n" + "=" * 70)
    print(f"Total selected: {len(selected)}")
    print("=" * 70)

    # Save with metadata
    out = {
        "seed": args.seed,
        "per_tercile": k,
        "selection_method": ("tercile-stratified by argument count, "
                             "fixed seed, from full 71-policy graph"),
        "tercile_ranges": {
            "sparse": count_range(sparse),
            "medium": count_range(medium),
            "dense":  count_range(dense),
        },
        "selected": [
            {"policy": p,
             "n_args": counts[p],
             "tercile": ("sparse" if p in sel_sparse else
                         "medium" if p in sel_medium else "dense")}
            for p in selected
        ],
        "n_selected": len(selected),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}")

    # Drop-in list
    print(f"\n" + "=" * 70)
    print("COPY THIS INTO policies.py")
    print("=" * 70)
    print("POLICIES = [")
    for p in selected:
        safe = p.replace('"', '\\"')
        print(f'    "{safe}",')
    print("]")


if __name__ == "__main__":
    main()