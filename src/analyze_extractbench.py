"""Analyze ExtractBench short-split results.

Reads results/<model>_extractbench-short.jsonl and reports per-model
P/R/F1 overall, by repetition stratum (occurrence-only: the ExtractBench
loader has no field_stats, so 'conflicting' is never assigned — see the
loader docstring), and STP rate. Gold counts per doc come from
load_extractbench(252).
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from step1_multipage_smoke import load_extractbench  # noqa: E402

MODELS = {
    "mini": "openrouter_openai_gpt-5-mini",
    "haiku": "openrouter_anthropic_claude-haiku-4.5",
    "flash": "openrouter_google_gemini-2.5-flash",
    "qwen": "openrouter_qwen_qwen3-235b-a22b-2507",
}


def main():
    docs = load_extractbench(252)
    gold_per_doc = {d["doc_id"]: len(d["gold"]) for d in docs}
    ngold = sum(gold_per_doc.values())
    print(f"docs loaded: {len(docs)} gold fields: {ngold}")
    print()
    print(f"{'model':<8}{'P':>7}{'R':>7}{'F1':>7}{'STP':>9}{'single':>9}{'ident':>8}")
    for short, prefix in MODELS.items():
        path = f"results/{prefix}_extractbench-short.jsonl"
        if not os.path.exists(path):
            print(f"{short:<8}MISSING ({path})")
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        tp = sum(1 for r in rows if r["correct"])
        npred = len(rows)
        p = tp / npred if npred else 0
        rc = tp / ngold if ngold else 0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        corr = Counter(r["doc"] for r in rows if r["correct"])
        stp = sum(1 for d, g in gold_per_doc.items() if corr.get(d, 0) == g)
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            by[r["stratum"]][0] += 1
            by[r["stratum"]][1] += int(bool(r["correct"]))
        s_acc = by["single"][1] / by["single"][0] if by["single"][0] else 0
        i_acc = by["repeated-identical"][1] / by["repeated-identical"][0] if by["repeated-identical"][0] else 0
        print(f"{short:<8}{p:>7.3f}{rc:>7.3f}{f1:>7.3f}{stp}/{len(docs):>6}"
              f"{s_acc:>9.2f}{i_acc:>8.2f}  (attempted {npred}, single {by['single'][0]}, ident {by['repeated-identical'][0]})")


if __name__ == "__main__":
    main()
