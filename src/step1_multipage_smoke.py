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
  cuad (default): canonical SQuAD-style JSON from the authors' GitHub repo
      (The-Atticus-Project/cuad, data.zip -> train_separate_questions.json,
      408 train docs; test.json, 102 test docs). Uses ~10 short-value
      labels; long clauses are excluded and the exclusion is reported,
      per design v2 §4. (HF theatticusproject/cuad no longer ships a
      loading script or annotations as of 2026-11-22 — do not use
      load_dataset for CUAD.)
  kleister: local clone at data/kleister-charity (see step1-setup-v2.md).
      Value-only TSV + OCR text (in.tsv.xz). Train 1,729 docs;
      split=test uses dev-0 (440 docs) because test-A labels are not
      shipped. Distinct values for the money fields use the same-line
      amounts heuristic; other fields report distinct 1.
  extractbench: HF llamaindex/ExtractBench, split "short", test-only.
      The dataset ships no text column: PDFs are downloaded to
      data/extractbench/docs/short/ and text is extracted with pypdf
      (cached as .txt). 252 short docs, median 30 gold fields/doc.

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
    """Normalise a value for occurrence counting / matching (design v2 §5).

    Lowercase; underscores are word separators; strip $, £, commas and
    whitespace; a parenthesised numeric value is negative, e.g. "(1,000)"
    -> "-1000"; a trailing all-zero decimal is dropped, e.g. "103373.00"
    -> "103373", so the same amount written with or without pence matches.
    Non-numeric parenthesised text just loses the parens, as before.
    """
    s = str(s).lower().replace("_", " ")
    s = s.replace("$", "").replace("£", "").replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 2 and s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if re.fullmatch(r"-?[\d\s.,]+", inner):
            return "-" + norm(inner)
        s = inner
    s = s.strip("()").strip()
    if re.fullmatch(r"-?\d+\.\d+", s):
        whole, frac = s.split(".")
        if set(frac) == {"0"}:
            s = whole
    return s


def unescape_kleister_text(t):
    """Undo the Kleister in.tsv escape sequences (README: \\n end of line,
    \\t tab, \\f page break, \\\\ literal backslash)."""
    return (
        t.replace("\\\\", "\x00")
        .replace("\\n", "\n")
        .replace("\\t", " ")
        .replace("\\f", "\n")
        .replace("\x00", "\\")
    )


# A monetary amount that looks like money, not a year or a count: it must
# carry a currency symbol, a thousands separator, or a decimal part.
_MONEY_RE = re.compile(r"£\s*\d[\d,]*(\.\d{1,2})?|\d{1,3}(,\d{3})+(\.\d{1,2})?|\d+\.\d{2}")


def kleister_money_distinct(gold_value, text):
    """Distinct monetary values on the same OCR lines as the gold amount.

    Kleister ground truth is value-only (one value per field per doc), so
    there are no annotated alternative values. The conflict signal here
    is heuristic: amounts printed on the same line as the gold amount are
    usually the prior-year / restated / subtotal figure in the same
    financial table. Returns 1 when no competing amount shares a line
    with the gold amount. Only used for the income/spending fields.
    """
    gold = norm(gold_value)
    if not gold:
        return 1
    distinct = set()
    for line in text.split("\n"):
        amounts = {norm(m.group(0)) for m in _MONEY_RE.finditer(line)}
        if gold in amounts:
            distinct |= amounts
    return max(1, len(distinct))


def count_occurrences(value, text):
    """Count normalised occurrences of `value` in `text` (substring count)."""
    v = norm(value)
    if not v or len(v) < 2:
        return 0
    return norm(text).count(v)


def stratum(n_occurrences, n_distinct_values, n_annotated=1):
    """Repetition stratum (design v2 §5) from three counts.

    n_occurrences: substring occurrences of the gold value in the doc text.
    n_distinct_values: distinct normalised values for the field in the doc.
    n_annotated: annotated occurrences of the field in the doc.
    Conflicting means the doc holds different values for the same field,
    even if the gold string itself occurs once; identical means the value
    recurs (in text or in annotations) with no competing value.
    """
    if n_distinct_values >= 2:
        return "repeated-conflicting"
    if n_occurrences >= 2 or n_annotated >= 2:
        return "repeated-identical"
    return "single"


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


def load_cuad(n, split="train"):
    # NOTE (2026-11-22): HF theatticusproject/cuad no longer ships a loading
    # script or annotation JSONs (only 511 PDFs), so `load_dataset` fails
    # split verification (expects 84,325 examples, gets 511). Load the
    # canonical SQuAD-style JSON from the authors' GitHub repo instead:
    # The-Atticus-Project/cuad, data.zip -> train_separate_questions.json
    # (408 train docs) and test.json (102 test docs). Question text embeds
    # the label as: related to "LABEL" ...; one gold value per label per
    # doc (first answer); labels with no answer (is_impossible) are skipped,
    # matching the old HF flatten_cuad semantics.
    # Distinct values (design v2 §5): all annotated answers for the label
    # in the doc, normalised; distinct > 1 means the doc holds different
    # values for the same field. Returned as doc["distinct"][label].
    # Caveat: format variants of the same value (e.g. "L.L.C." vs "LLC",
    # "September 7, 1999" vs "7th day of September, 1999") count as
    # distinct under the §5 normalisation, so the conflicting share is an
    # upper bound for genuinely different values.
    paths = {
        "train": ("data/train_separate_questions.json", "cuad-train-"),
        "test": ("data/test.json", "cuad-test-"),
    }
    if split not in paths:
        raise ValueError(f"unknown CUAD split {split!r}, want train or test")
    json_path, prefix = paths[split]
    if not os.path.exists(json_path):
        print(f"CUAD JSON not found at {json_path}.")
        print("Download it from the authors' repo:")
        print("  curl -L -o data/data.zip https://github.com/The-Atticus-Project/cuad/raw/main/data.zip")
        print("  cd data && unzip -o data.zip train_separate_questions.json test.json CUADv1.json")
        raise SystemExit(2)
    with open(json_path) as fh:
        raw = json.load(fh)
    docs = []
    for i, entry in enumerate(raw["data"][:n]):
        text = " ".join(p.get("context", "") for p in entry["paragraphs"])
        pairs = []
        seen = set()
        all_answers = {}
        for para in entry["paragraphs"]:
            for qa in para["qas"]:
                m = re.search(r'related to "([^"]+)"', qa["question"])
                if not m:
                    continue
                label = m.group(1)
                if label not in CUAD_SHORT_LABELS:
                    continue
                for a in qa.get("answers") or []:
                    if a.get("text"):
                        all_answers.setdefault(label, []).append(a["text"])
                if label in seen:
                    continue
                if not qa.get("answers"):
                    continue
                value = qa["answers"][0]["text"]
                if value:
                    seen.add(label)
                    pairs.append((label, value))
        if not text or not pairs:
            continue
        distinct = {
            label: {
                "distinct": max(1, len({norm(v) for v in vals})),
                "n_annot": len(vals),
            }
            for label, vals in all_answers.items()
        }
        docs.append({
            "doc_id": f"{prefix}{i}",
            "text": text,
            "gold": pairs,
            "field_stats": distinct,
        })
    return docs


KLEISTER_KEYS = [
    "address__post_town",
    "address__postcode",
    "address__street_line",
    "charity_name",
    "charity_number",
    "income_annually_in_british_pounds",
    "report_date",
    "spending_annually_in_british_pounds",
]

KLEISTER_MONEY_KEYS = {
    "income_annually_in_british_pounds",
    "spending_annually_in_british_pounds",
}


def load_kleister(n, split="train"):
    """Load Kleister Charity docs from the local clone at data/kleister-charity.

    Layout (README): train/ (1,729 docs, expected.tsv present),
    dev-0/ (440 docs, expected.tsv present), test-A/ (609 docs, expected.tsv
    NOT shipped — hidden). So split="test" uses dev-0 as the held-out test,
    per design v2 §4 fallback. Text is the combined OCR column (last of the
    6 in.tsv.xz columns): djvu text unless tesseract is 1000+ chars longer.
    Gold values come from expected.tsv (key=value pairs, underscores stand
    for spaces, some fields upper-cased); expected-original.tsv holds the
    true-cased values. Text is NOT the PDF: documents/ holds git-annex
    symlinks and the PDFs were not downloaded (text-only work does not
    need them). Distinct values: Kleister ground truth is value-only, so
    for the money fields the conflict signal is the same-line heuristic
    in kleister_money_distinct; every other field reports distinct 1.
    """
    import lzma

    dirs = {"train": "train", "test": "dev-0"}
    if split not in dirs:
        raise ValueError(f"unknown Kleister split {split!r}, want train or test")
    data_dir = os.path.join("data", "kleister-charity", dirs[split])
    prefix = f"kleister-{split}-"
    in_path = os.path.join(data_dir, "in.tsv.xz")
    exp_path = os.path.join(data_dir, "expected.tsv")
    if not os.path.isdir(data_dir) or not os.path.exists(in_path):
        print(f"Kleister data not found at {data_dir}.")
        print("Run: git clone https://github.com/applicaai/kleister-charity.git data/kleister-charity")
        print("Then re-run with --dataset kleister. (No API spend in --dry-run.)")
        raise SystemExit(2)
    if not os.path.exists(exp_path):
        print(f"Kleister expected.tsv not found at {exp_path}.")
        print("Test annotations are not shipped for test-A; use --split test (dev-0) instead.")
        raise SystemExit(2)
    with open(exp_path, encoding="utf-8") as fh:
        expected = [ln.rstrip("\n") for ln in fh]
    docs = []
    with lzma.open(in_path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n or i >= len(expected):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            text = unescape_kleister_text(parts[5])
            pairs = []
            field_stats = {}
            for kv in expected[i].split(" "):
                if "=" not in kv:
                    continue
                key, value = kv.split("=", 1)
                if key not in KLEISTER_KEYS or not value:
                    continue
                pairs.append((key, value.replace("_", " ")))
                if key in KLEISTER_MONEY_KEYS:
                    field_stats[key] = {
                        "distinct": kleister_money_distinct(value, text),
                        "n_annot": 1,
                    }
                else:
                    field_stats[key] = {"distinct": 1, "n_annot": 1}
            if not text or not pairs:
                continue
            docs.append({
                "doc_id": f"{prefix}{i}",
                "text": text,
                "gold": pairs,
                "field_stats": field_stats,
            })
    return docs


def load_extractbench(n):
    """Load ExtractBench short split (test-only, 252 docs).

    The HF dataset ships NO document text column — only a `pdf` path
    (docs/short/....pdf) plus JSON schema / expected_output / field_rules.
    Text is extracted locally from PDFs downloaded into
    data/extractbench/ (huggingface_hub snapshot_download, docs/short/*.pdf
    only) with pypdf, cached as .txt next to each PDF so re-runs are free.
    Docs whose PDF yields no extractable text are skipped and reported
    by the dry-run count. Caveat: expected_output is flattened one level
    (nested dict/list values become str(v) Python reprs, ~4% of gold —
    effectively unwinnable under exact match; recursive flattening is a
    Week 1 scoring decision, not changed here).
    """
    import pypdf

    from datasets import load_dataset

    ds = load_dataset("llamaindex/ExtractBench", split="short")
    docs = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        expected = ex.get("expected_output") or {}
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                expected = {}
        pairs = [(k, str(v)) for k, v in expected.items() if v not in (None, "")]
        if not pairs:
            continue
        pdf_rel = ex.get("pdf") or ""
        pdf_path = os.path.join("data", "extractbench", pdf_rel)
        text = ""
        if os.path.exists(pdf_path):
            txt_path = pdf_path[:-4] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as fh:
                    text = fh.read()
            else:
                try:
                    reader = pypdf.PdfReader(pdf_path)
                    text = "\n".join((p.extract_text() or "") for p in reader.pages)
                    with open(txt_path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                except Exception:
                    text = ""
        if text and pairs:
            docs.append({"doc_id": f"extractbench-short-{i}", "text": text, "gold": pairs})
    return docs


def dry_run(docs):
    strata = Counter()
    total_gold = 0
    for doc in docs:
        stats = doc.get("field_stats", {})
        norm_text = norm(doc["text"])
        for key, value in doc["gold"]:
            total_gold += 1
            n_occ = norm_text.count(norm(value)) if norm(value) and len(norm(value)) >= 2 else 0
            fs = stats.get(key, {"distinct": 1, "n_annot": 1})
            strata[stratum(n_occ, fs["distinct"], fs["n_annot"])] += 1
    print(f"docs={len(docs)} gold_fields={total_gold}")
    print(f"strata (distinct values counted, design v2 §5): {dict(strata)}")
    lens = [len(d["text"]) for d in docs]
    if lens:
        print(f"text chars per doc: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
    print("Dry run done — no API calls made.")


def parse_json(text):
    # Parse the first JSON object in the response. raw_decode stops at the
    # end of that object, so trailing prose (which broke the old greedy
    # regex when it contained a '}') and ``` fences are tolerated.
    # A truncated response still returns [] (caller retries once).
    if not text:
        return []
    start = text.find("{")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj.get("fields", []) if isinstance(obj, dict) else []
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def run_llm(docs, model, max_chars=12000, out_path=None, append=False, delay=0):
    from dotenv import load_dotenv

    import litellm

    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("FAIL: OPENROUTER_API_KEY not set. See step1-setup-v2.md.")
        raise SystemExit(1)
    os.makedirs("results", exist_ok=True)
    if out_path is None:
        out_path = f"results/{model.replace('/', '_')}_multipage_smoke.jsonl"
    tp = npred = ngold = 0
    cost = 0.0
    lat = []
    with open(out_path, "a" if append else "w") as fout:
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
                print(f"{doc['doc_id']}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
                if delay:
                    time.sleep(delay)
                continue
            lat.append(time.time() - t0)
            fields = parse_json(text)
            if not fields:
                # One retry: a model sometimes returns truncated JSON or
                # trailing prose that the parser rejects (haiku did this on
                # ~5% of CUAD test docs). Retry once before counting the doc
                # as unanswered.
                try:
                    r2 = litellm.completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text2 = r2.choices[0].message.content
                    cost += litellm.completion_cost(completion_response=r2) or 0.0
                    fields = parse_json(text2)
                    print(f"{doc['doc_id']}: retry parsed {len(fields)} fields", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{doc['doc_id']}: retry ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
            for f in fields:
                key, val = f.get("key", ""), f.get("value", "")
                item = (key, norm(val))
                ok = gold[item] > 0
                if ok:
                    gold[item] -= 1
                    tp += 1
                npred += 1
                gold_vals = [v for k, v in doc["gold"] if k == key]
                n_occ = max((count_occurrences(v, doc["text"]) for v in gold_vals), default=0)
                fs = doc.get("field_stats", {}).get(key, {"distinct": 1, "n_annot": 1})
                fout.write(json.dumps({
                    "doc": doc["doc_id"],
                    "key": key,
                    "pred": val,
                    "correct": ok,
                    "confidence": f.get("confidence"),
                    "gold_occurrences": n_occ,
                    "stratum": stratum(n_occ, fs["distinct"], fs["n_annot"]),
                }, ensure_ascii=False) + "\n")
            print(f"{doc['doc_id']}: ${cost:.4f} total so far, {lat[-1]:.1f}s", flush=True)
            if delay:
                time.sleep(delay)
    p = tp / npred if npred else 0
    rcl = tp / ngold if ngold else 0
    f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0
    print(f"\nmodel={model} docs={len(lat)} P={p:.3f} R={rcl:.3f} F1={f1:.3f}")
    print(f"total cost=${cost:.4f} avg latency={sum(lat)/max(len(lat), 1):.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cuad", "kleister", "extractbench"], default="cuad")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="CUAD: train (408) or test (102). Kleister: train (1729) or test (dev-0, 440 — test-A labels are not shipped).")
    ap.add_argument("--model", default="openrouter/openai/gpt-5-mini")
    ap.add_argument("--out", default=None,
                    help="Results JSONL path. Default results/<model>_multipage_smoke.jsonl.")
    ap.add_argument("--run-llm", action="store_true", help="Call the model (spends API credit). Default is dry-run.")
    ap.add_argument("--dry-run", action="store_true", help="No API calls (default).")
    ap.add_argument("--start", type=int, default=0,
                    help="Resume: skip the first N docs (use with --append).")
    ap.add_argument("--append", action="store_true",
                    help="Append rows to --out instead of overwriting (resume mode).")
    ap.add_argument("--delay", type=float, default=0,
                    help="Seconds to sleep after each doc (rate-limit pacing).")
    args = ap.parse_args()

    loaders = {"cuad": load_cuad, "kleister": load_kleister, "extractbench": load_extractbench}
    if args.dataset == "extractbench":
        docs = load_extractbench(args.n)
    else:
        docs = loaders[args.dataset](args.n, split=args.split)
    if not docs:
        print(f"No docs loaded for --dataset {args.dataset}.")
        return
    if args.start:
        docs = docs[args.start:]
    dry_run(docs)
    if args.run_llm:
        run_llm(docs, args.model, out_path=args.out, append=args.append, delay=args.delay)


if __name__ == "__main__":
    main()
