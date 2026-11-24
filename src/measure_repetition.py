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


def distinct_values(value, text, distinct=1):
    """Distinct-value count for the conflicting stratum.

    The loaders now supply it: CUAD counts distinct normalised annotated
    answers for the label in the doc; Kleister counts distinct monetary
    amounts sharing a line with the gold amount (money fields only).
    This helper is kept for callers that only have a raw value and text.
    """
    return distinct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cuad", "kleister"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from step1_multipage_smoke import load_kleister, norm

    loaders = {"cuad": load_cuad, "kleister": load_kleister}
    docs = loaders[args.dataset](args.n, split=args.split)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    strata = Counter()
    by_key = {}
    with open(args.out, "w") as fout:
        for doc in docs:
            stats = doc.get("field_stats", {})
            norm_text = norm(doc["text"])
            for key, value in doc["gold"]:
                n_occ = norm_text.count(norm(value)) if norm(value) and len(norm(value)) >= 2 else 0
                fs = stats.get(key, {"distinct": 1, "n_annot": 1})
                s = stratum(n_occ, fs["distinct"], fs["n_annot"])
                strata[s] += 1
                by_key.setdefault(key, Counter())[s] += 1
                fout.write(json.dumps({
                    "doc": doc["doc_id"],
                    "key": key,
                    "value": value,
                    "occurrences": n_occ,
                    "distinct": fs["distinct"],
                    "n_annot": fs["n_annot"],
                    "stratum": s,
                }, ensure_ascii=False) + "\n")
    print(f"dataset={args.dataset} split={args.split} docs={len(docs)}")
    print(f"strata (distinct values counted, design v2 §5): {dict(strata)}")
    for key in sorted(by_key):
        print(f"  {key}: {dict(by_key[key])}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
