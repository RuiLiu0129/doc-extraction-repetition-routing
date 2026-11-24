"""Analyze the injected-conflict probe results.

Reads results/probe_injected_conflict_manifest.json, the probe JSONL files
(results/<model>_probe_<arm>.jsonl), and the Batch 1 CUAD test102 files
(results/<model>_test102.jsonl) for the control (original text) accuracy
on the same docs + field.

Per model x arm (conflicting / identical), reports on the TARGET field only:
  n docs, correct (model output the gold value), picked_injected
  (conflicting arm: model output the injected value), missed (no prediction
  for the field), wrong_other. Plus overall P/R/F1 and STP for the arm's
  full run (all fields), for comparability with the Batch 1/2 tables.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from step1_multipage_smoke import norm  # noqa: E402

MODELS = {
    "mini": "openrouter_openai_gpt-5-mini",
    "haiku": "openrouter_anthropic_claude-haiku-4.5",
    "flash": "openrouter_google_gemini-2.5-flash",
    "qwen": "openrouter_qwen_qwen3-235b-a22b-2507",
}
ARMS = ["conflicting", "identical"]


def load_rows(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    man = json.load(open("results/probe_injected_conflict_manifest.json", encoding="utf-8"))
    by_src = {m["source_doc"]: m for m in man}
    print(f"probe docs: {len(man)}")
    print()
    print("Per-model target-field accuracy (n=50 each):")
    print(f"{'model':<8}{'control':>10}{'conflicting':>13}{'picked_inj':>11}{'identical':>11}")
    for short, prefix in MODELS.items():
        # control: Batch 1 CUAD test102 rows for the same doc + field
        ctrl = load_rows(f"results/{prefix}_test102.jsonl")
        ctrl_map = {}
        if ctrl:
            for r in ctrl:
                ctrl_map[(r["doc"], r["key"])] = r["correct"]
        ctrl_ok = sum(1 for m in man if ctrl_map.get((m["source_doc"], m["field"])) is True)
        row = [short, f"{ctrl_ok}/50"]
        for arm in ARMS:
            rows = load_rows(f"results/{prefix}_probe_{arm}.jsonl")
            if rows is None:
                row.append("MISSING")
                if arm == "conflicting":
                    row.append("-")
                continue
            by_doc = {}
            for r in rows:
                by_doc.setdefault(r["doc"], []).append(r)
            correct = picked = missed = 0
            for m in man:
                doc_id = f"probe-{arm}-{m['source_doc']}"
                rs = [r for r in by_doc.get(doc_id, []) if r["key"] == m["field"]]
                if not rs:
                    missed += 1
                    continue
                if any(r["correct"] for r in rs):
                    correct += 1
                elif arm == "conflicting" and any(
                        norm(r["pred"]) == norm(m["injected_value"]) for r in rs):
                    picked += 1
            row.append(f"{correct}/50")
            if arm == "conflicting":
                row.append(f"{picked}/50")
        print(f"{row[0]:<8}{row[1]:>10}{row[2]:>13}{row[3]:>11}{row[4]:>11}")
    print()
    print("Full-run P/R/F1 + STP per arm (all fields, for reference):")
    for arm in ARMS:
        for short, prefix in MODELS.items():
            rows = load_rows(f"results/{prefix}_probe_{arm}.jsonl")
            if rows is None:
                print(f"{arm} {short}: MISSING")
                continue
            # gold per doc from manifest source docs: reload probe docs
            docs = json.load(open("results/probe_docs.json", encoding="utf-8"))[arm]
            ngold = sum(len(d["gold"]) for d in docs)
            tp = sum(1 for r in rows if r["correct"])
            npred = len(rows)
            p = tp / npred if npred else 0
            rc = tp / ngold if ngold else 0
            f1 = 2 * p * rc / (p + rc) if p + rc else 0
            # STP: docs where correct == gold count
            from collections import Counter
            corr = Counter(r["doc"] for r in rows if r["correct"])
            stp = sum(1 for d in docs if corr.get(d["doc_id"], 0) == len(d["gold"]))
            print(f"{arm:<12}{short:<8}P={p:.3f} R={rc:.3f} F1={f1:.3f} STP={stp}/{len(docs)}")


if __name__ == "__main__":
    main()
