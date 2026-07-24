"""
calibrate_contradicts.py
========================
Pick CONTRADICTS thresholds from EVIDENCE before paying for a full
regeneration. Runs the (patched) bidirectional NLI scoring on a SMALL
subset of policies, then reports — without writing any edges — what
each candidate threshold setting would do to selectivity / saturation.

WHY THIS EXISTS
---------------
The current graph is saturated (~33% of all pro x con pairs became
CONTRADICTS) because acceptance was single-direction P(contra) >= 0.60,
and that score is ~0.999 for nearly any opposing-stance pair. The
proposed fix is (a) bidirectional min-contradiction and (b) a higher
cosine blocking floor. But it is unknown whether min(fwd,bwd) is itself
still ~0.999. This harness measures that on real data cheaply, so the
full run is done ONCE with thresholds that are known to work.

WHAT IT DOES
------------
  1. Loads the export, picks N_POLICIES (default 3) spanning the
     saturation range (most / median / least saturated, if you pass
     --saturation-csv from diagnose_edges.py; otherwise first N).
  2. Blocks cross-stance pairs at a LOW floor (0.40) so you can see the
     full distribution and test higher floors post-hoc.
  3. Scores every candidate bidirectionally (contra_fwd, contra_bwd).
  4. Prints distributions of: cosine, contra_fwd, contra_min=min(fwd,bwd),
     and the GAP (fwd - bwd) that reveals how asymmetric contradiction is.
  5. Sweeps a grid of (cosine_floor, tau_contra_min) and prints, per
     setting: edges kept, and saturation = kept / possible_cross_pairs
     for the sampled policies. THIS is the table you choose from.

USAGE
-----
  python calibrate_contradicts.py --export data/neo4j_export_71full.json
  python calibrate_contradicts.py --export ... --policies 3
  python calibrate_contradicts.py --export ... --saturation-csv saturation.csv
  python calibrate_contradicts.py --export ... --policy-names "We should legalize cannabis" "We should ban whaling"
"""

import os
import sys
import json
import argparse
import numpy as np

from common import load_json, nli_label_batch

LOW_FLOOR = 0.40   # block low so the full distribution is visible


def build_arg_index(export):
    args = export.get("arguments", {})
    texts, embs = {}, {}
    for aid, a in args.items():
        emb = a.get("embedding")
        txt = (a.get("text") or "").strip()
        if emb and txt:
            v = np.asarray(emb, dtype=float)
            n = np.linalg.norm(v)
            if n > 0:
                texts[aid] = txt
                embs[aid] = v / n
    return texts, embs


def dist(name, xs):
    if not xs:
        print(f"  {name:<16} (empty)")
        return
    xs = sorted(xs)
    n = len(xs)
    def q(p): return xs[min(n-1, int(p*n))]
    print(f"  {name:<16} n={n:<7} min={xs[0]:.3f} p10={q(.10):.3f} "
          f"p25={q(.25):.3f} med={q(.50):.3f} p75={q(.75):.3f} "
          f"p90={q(.90):.3f} max={xs[-1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="data/neo4j_export_71full.json")
    ap.add_argument("--policies", type=int, default=3,
                    help="how many policies to sample if names not given")
    ap.add_argument("--policy-names", nargs="*", default=None,
                    help="explicit policy names to calibrate on")
    ap.add_argument("--saturation-csv", default=None,
                    help="diagnose_edges.py CSV; picks high/med/low sat policies")
    args = ap.parse_args()

    if not os.path.exists(args.export):
        print(f"[STOP] not found: {args.export}")
        sys.exit(1)

    export = load_json(args.export)
    texts, embs = build_arg_index(export)
    policies = export.get("policies", {})

    # ---- choose policies ----
    if args.policy_names:
        chosen = [p for p in args.policy_names if p in policies]
    elif args.saturation_csv and os.path.exists(args.saturation_csv):
        import csv
        rows = []
        with open(args.saturation_csv) as f:
            for r in csv.DictReader(f):
                rows.append((r["policy"], float(r["saturation"])))
        rows.sort(key=lambda x: x[1], reverse=True)
        # most, median, least saturated
        idx = [0, len(rows)//2, len(rows)-1]
        chosen = [rows[i][0] for i in idx if rows[i][0] in policies]
    else:
        chosen = list(policies.keys())[:args.policies]

    print("=" * 64)
    print("CALIBRATION — bidirectional CONTRADICTS on sampled policies")
    print("=" * 64)
    print("Policies:")
    for p in chosen:
        print(f"  - {p}")
    print()

    # ---- block + score each chosen policy ----
    all_cos, all_cf, all_cmin, all_gap = [], [], [], []
    # keep per-pair records to do the threshold sweep
    records = []   # (cosine, contra_min)
    possible_cross = 0

    for policy in chosen:
        data = policies[policy]
        pro_ids = [a for a in data.get("pros", []) if a in embs]
        con_ids = [a for a in data.get("cons", []) if a in embs]
        if not pro_ids or not con_ids:
            continue
        possible_cross += len(pro_ids) * len(con_ids)

        P = np.stack([embs[a] for a in pro_ids])
        C = np.stack([embs[a] for a in con_ids])
        sims = P @ C.T
        pi, ci = np.where(sims >= LOW_FLOOR)
        pairs = [(pro_ids[p], con_ids[c], float(sims[p, c]))
                 for p, c in zip(pi, ci)]
        if not pairs:
            continue

        print(f"  scoring {len(pairs):,} blocked pairs for: {policy[:45]} ...")
        fp = [texts[a] for (a, b, _) in pairs]
        fh = [texts[b] for (a, b, _) in pairs]
        bp = [texts[b] for (a, b, _) in pairs]
        bh = [texts[a] for (a, b, _) in pairs]

        BATCH = 64
        cf_all, cb_all = [], []
        for i in range(0, len(pairs), BATCH):
            fpr = nli_label_batch(fp[i:i+BATCH], fh[i:i+BATCH], batch_size=BATCH)
            bpr = nli_label_batch(bp[i:i+BATCH], bh[i:i+BATCH], batch_size=BATCH)
            cf_all.extend(p[0] for p in fpr)   # index 0 = contradiction
            cb_all.extend(p[0] for p in bpr)

        for (a, b, cos), cf, cb in zip(pairs, cf_all, cb_all):
            cmin = min(cf, cb)
            all_cos.append(cos)
            all_cf.append(cf)
            all_cmin.append(cmin)
            all_gap.append(abs(cf - cb))
            records.append((cos, cmin))

    # ---- distributions ----
    print("\n" + "=" * 64)
    print("  DISTRIBUTIONS (blocked candidates, floor=0.40)")
    print("=" * 64)
    dist("cosine", all_cos)
    dist("contra_fwd", all_cf)
    dist("contra_min", all_cmin)
    dist("|fwd - bwd|", all_gap)

    print("\n  Reading the gap: if |fwd-bwd| is almost always ~0, contradiction")
    print("  is symmetric and bidirectional-min will NOT thin the graph — the")
    print("  cosine floor becomes the main lever. If the gap is often large,")
    print("  bidirectional-min is doing real work.")

    # ---- threshold sweep ----
    print("\n" + "=" * 64)
    print("  SATURATION SWEEP  (kept / possible cross-stance pairs)")
    print(f"  possible cross-stance pairs in sample: {possible_cross:,}")
    print("=" * 64)
    cos_floors = [0.50, 0.60, 0.70, 0.80]
    tau_mins   = [0.60, 0.80, 0.90, 0.95, 0.99]
    header = "  cos_floor \\ tau_min " + "".join(f"{t:>8}" for t in tau_mins)
    print(header)
    print("  " + "-" * (len(header)-2))
    for cf_floor in cos_floors:
        cells = []
        for tau in tau_mins:
            kept = sum(1 for (cos, cmin) in records
                       if cos >= cf_floor and cmin >= tau)
            sat = kept / possible_cross if possible_cross else 0.0
            cells.append(f"{100*sat:6.1f}%")
        print(f"  cos>={cf_floor:<14}" + "".join(f"{c:>8}" for c in cells))

    print("\n  TARGET: pick the (cos_floor, tau_min) whose saturation lands")
    print("  in a defensible rebuttal range — roughly 5-15%. Then set")
    print("  FLOOR_CONTRA and TAU_CONTRA in generate_relations.py to match")
    print("  and run the full regeneration ONCE.")


if __name__ == "__main__":
    main()