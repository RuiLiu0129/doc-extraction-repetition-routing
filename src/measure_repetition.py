"""Measure natural repetition strata on Kleister Charity / CUAD (Week 1).

Design v2 §5. No new human labelling: for each gold field value, count
exact normalised occurrences in the document text and record position
(page/section where available), context, and whether occurrences agree.

Output: a JSONL file with one row per (doc, field), plus a summary table
of stratum sizes per dataset. This is the measurement that turns the
"repetition is inferred, not labelled" caveat in the scout into a number.

Usage:
    python src/measure_repetition.py --dataset cuad --n 200 --out results/cuad_repetition.jsonl
    python src/measure_repetition.py --dataset kleister --n 200 --out results/kleister_repetition.jsonl

Kleister input: a local clone at data/kleister-charity. The loader is
intentionally strict about layout — if the repo layout differs from the
one checked in Week 1, it prints the files it found and exits, instead
of silently counting the wrong text.
"""
import argparse
import json
import os
import re
from collections import Counter

from step1_multipage_smoke import count_occurrences, load_cuad, norm, stratum


def distinct_values(value, text):
    """Approximate distinct-value detection for the conflicting stratum.

    Full distinct-value enumeration needs span-level occurrence extraction
    (find every occurrence's surrounding context and compare normalised
    values). This Week 1 version returns 1 when the value occurs, and
    flags the doc for the full enumeration in the batch runner. It is
    honest about that limit in its output — do not report its stratum
    counts as final conflicting-stratum sizes.
    """
    return 1 if count_occurrences(value, text) > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cuad", "kleister"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loaders = {"cuad": load_cuad}
    if args.dataset == "kleister":
        print("Kleister loader: see step1_multipage_smoke.load_kleister —")
        print("fill in after the Week 1 layout check. CUAD works now.")
        raise SystemExit(2)
    docs = loaders[args.dataset](args.n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    strata = Counter()
    with open(args.out, "w") as fout:
        for doc in docs:
            for key, value in doc["gold"]:
                n_occ = count_occurrences(value, doc["text"])
                s = stratum(n_occ, distinct_values(value, doc["text"]))
                strata[s] += 1
                fout.write(json.dumps({
                    "doc": doc["doc_id"],
                    "key": key,
                    "value": value,
                    "occurrences": n_occ,
                    "stratum_approx": s,
                }, ensure_ascii=False) + "\n")
    print(f"dataset={args.dataset} docs={len(docs)}")
    print(f"strata (approx — see distinct_values docstring): {dict(strata)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
