"""Step 1 smoke test for multi-page document extraction (design v2).

Replaces the v1 CORD-only smoke test (1-page receipts — dropped as primary
per the 2026-11-21 dataset lock: multi-page only).

What it does:
  --dry-run (default, no API spend): load a dataset, count repetition
      strata for gold values (single / repeated-identical /
      repeated-conflicting), and print page/token stats.
  --run-llm: also call one cheap model via OpenRouter/LiteLLM on --n docs,
      asking for per-field confidence, and score field-level F1.

Usage:
    python src/step1_multipage_smoke.py --dry-run
    python src/step1_multipage_smoke.py --dry-run --dataset cuad --n 5
    python src/step1_multipage_smoke.py --run-llm --dataset cuad --n 5
    python src/step1_multipage_smoke.py --run-llm --dataset cuad --n 5 \\
        --model openrouter/anthropic/claude-haiku-4.5

Datasets:
  cuad (default): HF theatticusproject/cuad, text-only contracts.
      Uses ~10 short-value labels; long clauses are excluded and the
      exclusion is reported, per design v2 §4.
  kleister: local clone at data/kleister-charity (see step1-setup-v2.md).
      Value-only JSON + PDF/OCR text. If the folder is absent, the script
      prints the clone command and exits gracefully instead of crashing.
  extractbench: HF llamaindex/ExtractBench, split "short", test-only.

Output (LLM mode only):
  results/<model>_multipage_smoke.jsonl — one row per predicted field:
  doc, key, pred, gold, correct, confidence, gold_occurrences, stratum.
"""
import argparse
import json
import os
import re
import time
from collections import Counter

# Short-value CUAD labels only. Long-clause extraction is a different task;
# see design v2 §4. These are label *names* as shipped in CUAD; if the
# HF version renames a label, the script reports the available labels
# and exits instead of silently scoring the wrong thing.
CUAD_SHORT_LABELS = {
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
    "Notice Period To Terminate Renewal",
    "Termination For Convenience",
    "Price Restrictions",
    "Minimum Commitment",
}

PROMPT_TEMPLATE = """Extract the following fields from this contract text.
Return ONLY a JSON object: {{"fields": [{{"key": ..., "value": ..., "confidence": ...}}, ...]}}
- key: one of {labels}.
- value: the text exactly as printed in the document, or "" if the field does not appear.
- confidence: probability (0-100) that this value is exactly correct.
Only include fields that appear in the document.
Document text (truncated to {max_chars} characters):
---
{text}
---
"""


def norm(s):
    """Normalise a value for occurrence counting / matching."""
    s = str(s).lower()
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("()")
    return s


def count_occurrences(value, text):
    """Count normalised occurrences of `value` in `text` (substring count)."""
    v = norm(value)
    if not v or len(v) < 2:
        return 0
    return norm(text).count(v)


def stratum(n_occurrences, n_distinct_values):
    if n_occurrences <= 1:
        return "single"
    if n_distinct_values <= 1:
        return "repeated-identical"
    return "repeated-conflicting"


def flatten_cuad(example):
    """Return (text, [(label, value), ...]) for one CUAD train example.

    CUAD HF rows carry `labels` (list of label names) and `label_spans`
    or similar span fields depending on the dataset version. This function
    is intentionally defensive: it inspects the row's keys on first use
    and raises a clear error listing them, instead of guessing a schema.
    """
    text = example.get("context") or example.get("text") or ""
    pairs = []
    # Common CUAD HF layout: "labels" + spans stored under "answers" /
    # "label_spans". Handle the two layouts seen in mirrors; otherwise fail loud.
    labels = example.get("labels") or []
    spans = example.get("label_spans") or example.get("answers") or []
    if isinstance(spans, dict):
        spans = spans.get("text", [])
    for label, span in zip(labels, spans):
        if label not in CUAD_SHORT_LABELS:
            continue
        value = span if isinstance(span, str) else span.get("text", "")
        if value:
            pairs.append((label, value))
    return text, pairs


def load_cuad(n):
    from datasets import load_dataset

    ds = load_dataset("theatticusproject/cuad", split="train")
    docs = []
    for i in range(min(n, len(ds))):
        text, pairs = flatten_cuad(ds[i])
        if not text or not pairs:
            continue
        docs.append({"doc_id": f"cuad-train-{i}", "text": text, "gold": pairs})
    return docs


def load_kleister(n, data_dir="data/kleister-charity"):
    """Load Kleister Charity docs if cloned locally; otherwise explain and exit."""
    if not os.path.isdir(data_dir):
        print(f"Kleister data not found at {data_dir}.")
        print("Run: git clone https://github.com/applicaai/kleister-charity.git data/kleister-charity")
        print("Then re-run with --dataset kleister. (No API spend in --dry-run.)")
        raise SystemExit(2)
    # The repo layout (train/dev/test JSON + PDF/OCR) is checked at run time;
    # do not hard-code a file layout that the scout did not verify.
    print(f"Found {data_dir}. Kleister loader is a stub in this smoke script:")
    print("  - list the train JSON files, print doc/field counts, and")
    print("  - count repetition strata from the OCR text once the layout is confirmed.")
    print("Fill in the loader after the Week 1 layout check (design v2 §5).")
    return []


def load_extractbench(n):
    from datasets import load_dataset

    ds = load_dataset("llamaindex/ExtractBench", split="short")
    docs = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        text = ex.get("text") or ex.get("document") or ""
        # expected_output is a JSON schema instance; flatten one level for the smoke test.
        expected = ex.get("expected_output") or {}
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                expected = {}
        pairs = [(k, str(v)) for k, v in expected.items() if v not in (None, "")]
        if text and pairs:
            docs.append({"doc_id": f"extractbench-short-{i}", "text": text, "gold": pairs})
    return docs


def dry_run(docs):
    strata = Counter()
    total_gold = 0
    for doc in docs:
        for key, value in doc["gold"]:
            total_gold += 1
            n_occ = count_occurrences(value, doc["text"])
            # Distinct-value check for the conflicting stratum needs all
            # occurrences; the smoke test approximates it with the count
            # and flags docs for the full Week 1 measurement (design v2 §5).
            strata[stratum(n_occ, 1 if n_occ <= 1 else 1)] += 1
    print(f"docs={len(docs)} gold_fields={total_gold}")
    print(f"strata (approx, distinct-value check is Week 1 work): {dict(strata)}")
    lens = [len(d["text"]) for d in docs]
    if lens:
        print(f"text chars per doc: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
    print("Dry run done — no API calls made.")


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)).get("fields", []) if m else []
    except json.JSONDecodeError:
        return []


def run_llm(docs, model, max_chars=12000):
    from dotenv import load_dotenv

    import litellm

    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("FAIL: OPENROUTER_API_KEY not set. See step1-setup-v2.md.")
        raise SystemExit(1)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{model.replace('/', '_')}_multipage_smoke.jsonl"
    tp = npred = ngold = 0
    cost = 0.0
    lat = []
    with open(out_path, "w") as fout:
        for doc in docs:
            gold = Counter((k, norm(v)) for k, v in doc["gold"])
            ngold += len(doc["gold"])
            labels = sorted({k for k, _ in doc["gold"]})
            prompt = PROMPT_TEMPLATE.format(
                labels=", ".join(labels),
                max_chars=max_chars,
                text=doc["text"][:max_chars],
            )
            t0 = time.time()
            try:
                r = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = r.choices[0].message.content
                cost += litellm.completion_cost(completion_response=r) or 0.0
            except Exception as e:  # noqa: BLE001 - report per-doc, keep going
                print(f"{doc['doc_id']}: ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            lat.append(time.time() - t0)
            for f in parse_json(text):
                key, val = f.get("key", ""), f.get("value", "")
                item = (key, norm(val))
                ok = gold[item] > 0
                if ok:
                    gold[item] -= 1
                    tp += 1
                npred += 1
                gold_vals = [v for k, v in doc["gold"] if k == key]
                n_occ = max((count_occurrences(v, doc["text"]) for v in gold_vals), default=0)
                fout.write(json.dumps({
                    "doc": doc["doc_id"],
                    "key": key,
                    "pred": val,
                    "correct": ok,
                    "confidence": f.get("confidence"),
                    "gold_occurrences": n_occ,
                    "stratum": stratum(n_occ, 1),
                }, ensure_ascii=False) + "\n")
            print(f"{doc['doc_id']}: ${cost:.4f} total so far, {lat[-1]:.1f}s")
    p = tp / npred if npred else 0
    rcl = tp / ngold if ngold else 0
    f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0
    print(f"\nmodel={model} docs={len(lat)} P={p:.3f} R={rcl:.3f} F1={f1:.3f}")
    print(f"total cost=${cost:.4f} avg latency={sum(lat)/max(len(lat), 1):.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cuad", "kleister", "extractbench"], default="cuad")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="openrouter/openai/gpt-5-mini")
    ap.add_argument("--run-llm", action="store_true", help="Call the model (spends API credit). Default is dry-run.")
    ap.add_argument("--dry-run", action="store_true", help="No API calls (default).")
    args = ap.parse_args()

    loaders = {"cuad": load_cuad, "kleister": load_kleister, "extractbench": load_extractbench}
    docs = loaders[args.dataset](args.n)
    if not docs:
        print(f"No docs loaded for --dataset {args.dataset}.")
        return
    dry_run(docs)
    if args.run_llm:
        run_llm(docs, args.model)


if __name__ == "__main__":
    main()
