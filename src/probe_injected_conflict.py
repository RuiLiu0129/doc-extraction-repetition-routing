"""Injected-conflict probe (design v2 §5) — build + run.

Paired design: take CUAD test docs where a date field (Agreement Date,
Effective Date, Expiration Date) occurs exactly once, and inject ONE
controlled perturbation per doc. Gold stays the original canonical value.
Two arms per doc:
  conflicting: the sentence containing the gold date is duplicated at ~75%
      through the document with the year changed (+1) and an "(As restated)"
      cue — the document now holds two different values for the same field.
  identical: the same sentence is duplicated verbatim (same value) — control
      for "does mere repetition hurt?", isolating conflict from repetition.
Control accuracy comes from the Batch 1 CUAD test102 results files
(results/openrouter_*_test102.jsonl) for the same docs + field — no
re-run of the original text is needed.

Manifest: results/probe_injected_conflict_manifest.json — one entry per
probe doc: source doc, arm, field, gold value, injected value (null for
identical), cue, injection char position, sentence used. The manifest +
this script make the probe reproducible; modified docs are NOT released
as a dataset (they are derived from CUAD text).

Usage:
  .venv/bin/python src/probe_injected_conflict.py --build [--n 50]
      Builds the probe docs + manifest (no API spend).
  .venv/bin/python src/probe_injected_conflict.py --run --arm conflicting --model openrouter/anthropic/claude-haiku-4.5 [--delay 4] [--out ...]
      Runs one model on one arm's probe docs (spends API credit; run
      through bin/run_with_key.py so the key is a surrogate).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from step1_multipage_smoke import (  # noqa: E402
    load_cuad,
    norm,
    run_llm,
)

TARGET_FIELDS = ["Agreement Date", "Effective Date", "Expiration Date"]
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MANIFEST_PATH = "results/probe_injected_conflict_manifest.json"
PROBE_DOCS_PATH = "results/probe_docs.json"
CHEAP_MODELS = [
    "openrouter/openai/gpt-5-mini",
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/qwen/qwen3-235b-a22b-2507",
]


def find_sentence(text, value):
    """Return (start, end, sentence) of the sentence containing value.

    Sentence boundaries are '. ' (CUAD text is space-joined paragraphs).
    Falls back to a 400-char window around the value if no boundary found.
    Returns None when the raw value string is not in the text.
    """
    idx = text.find(value)
    if idx < 0:
        return None
    start = text.rfind(". ", 0, idx)
    start = 0 if start < 0 else start + 2
    end = text.find(". ", idx + len(value))
    end = len(text) if end < 0 else end + 1
    return start, end, text[start:end]


def sentence_boundary_near(text, frac):
    """Char offset of a sentence boundary ('. ') nearest to frac*len(text)."""
    target = int(len(text) * frac)
    after = text.find(". ", target)
    before = text.rfind(". ", 0, target)
    if after < 0:
        return len(text)
    if before < 0:
        return after + 2
    return after + 2 if (after - target) <= (target - before) else before + 2


def build(n_target=50):
    docs = load_cuad(102, split="test")
    probe = []  # (source_doc, field, gold, sentence, s_start, s_end)
    skipped = {"no_single_date": 0, "no_year": 0, "no_sentence": 0,
               "orig_not_visible": 0, "inj_not_visible": 0}
    for doc in docs:
        if len(probe) >= n_target:
            break
        picked = None
        for field in TARGET_FIELDS:
            gold_vals = [v for k, v in doc["gold"] if k == field]
            if not gold_vals:
                continue
            gold = gold_vals[0]
            if norm(doc["text"]).count(norm(gold)) != 1:
                continue
            if not YEAR_RE.search(gold):
                skipped["no_year"] += 1
                continue
            sent = find_sentence(doc["text"], gold)
            if sent is None:
                skipped["no_sentence"] += 1
                continue
            picked = (field, gold, sent)
            break
        if picked is None:
            skipped["no_single_date"] += 1
            continue
        field, gold, (s0, s1, sentence) = picked
        # VISIBILITY (v2 fix, 2026-11-23): run_llm truncates doc text to
        # max_chars=12000, so an injection past 12k chars is invisible to
        # the model and the probe measures nothing. The original sentence
        # AND the injected sentence must both end before char 12000.
        # v1 injected at 75% of doc length: only 16/50 injections were
        # visible, so v1 results are superseded (files renamed _v1flawed).
        if s1 >= 12000:
            skipped["orig_not_visible"] += 1
            continue
        frac = min(0.75, 10000 / len(doc["text"]))
        pos = sentence_boundary_near(doc["text"], frac)
        if pos <= s1:
            nxt = doc["text"].find(". ", s1)
            pos = len(doc["text"]) if nxt < 0 else nxt + 2
        injected_len = len(sentence) + len("(As restated) ")
        if pos + injected_len >= 12000:
            skipped["inj_not_visible"] += 1
            continue
        probe.append((doc, field, gold, sentence, pos))
    print(f"candidate docs: {len(probe)} (target {n_target}); skipped: {skipped}")

    manifest = []
    out_docs = {"conflicting": [], "identical": []}
    for doc, field, gold, sentence, pos in probe:
        year_m = YEAR_RE.search(gold)
        old_year = int(year_m.group(0))
        new_gold = gold[: year_m.start()] + str(old_year + 1) + gold[year_m.end():]
        injected_sentence = "(As restated) " + sentence.replace(gold, new_gold, 1)
        # sanity: injected sentence must contain the new value
        assert new_gold in injected_sentence
        # pos was chosen visibility-aware in the selection loop above
        # (both sentences fully inside the first 12000 chars).
        # identical arm: verbatim duplicate of the sentence
        text_ident = doc["text"][:pos] + " " + sentence + doc["text"][pos:]
        # conflicting arm: duplicated sentence with changed year + cue
        text_conf = doc["text"][:pos] + " " + injected_sentence + doc["text"][pos:]
        src_id = doc["doc_id"]
        for arm, text in (("identical", text_ident), ("conflicting", text_conf)):
            fs = dict(doc.get("field_stats", {}))
            fs[field] = {"distinct": 2 if arm == "conflicting" else 1, "n_annot": 1}
            out_docs[arm].append({
                "doc_id": f"probe-{arm}-{src_id}",
                "text": text,
                "gold": doc["gold"],
                "field_stats": fs,
            })
        manifest.append({
            "source_doc": src_id,
            "field": field,
            "gold_value": gold,
            "injected_value": new_gold,
            "cue": "(As restated)",
            "injection_char_pos": pos,
            "doc_chars": len(doc["text"]),
            "sentence": sentence[:400],
        })
    os.makedirs("results", exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    with open(PROBE_DOCS_PATH, "w", encoding="utf-8") as fh:
        json.dump(out_docs, fh, ensure_ascii=False)
    print(f"probe docs built: {len(manifest)} per arm; manifest -> {MANIFEST_PATH}")
    print(f"fields used: { {f: sum(1 for m in manifest if m['field'] == f) for f in TARGET_FIELDS} }")
    return manifest


def run_arm(arm, model, delay=0, out=None, resume=False):
    with open(PROBE_DOCS_PATH, encoding="utf-8") as fh:
        docs = json.load(fh)[arm]
    if out is None:
        out = f"results/{model.replace('/', '_')}_probe_{arm}.jsonl"
    append = False
    if resume and os.path.exists(out):
        have = set()
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    have.add(json.loads(line)["doc"])
                except (json.JSONDecodeError, KeyError):
                    continue
        docs = [d for d in docs if d["doc_id"] not in have]
        append = True
        print(f"resume: {len(docs)} docs missing rows in {out}", flush=True)
    print(f"arm={arm} model={model} docs={len(docs)} -> {out} append={append}", flush=True)
    run_llm(docs, model, out_path=out, append=append, delay=delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--arm", choices=["conflicting", "identical"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--delay", type=float, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Run only docs missing from --out (appends).")
    args = ap.parse_args()
    if args.build:
        build(args.n)
    if args.run:
        if not args.arm or not args.model:
            print("--run needs --arm and --model")
            raise SystemExit(2)
        run_arm(args.arm, args.model, delay=args.delay, out=args.out,
                resume=args.resume)


if __name__ == "__main__":
    main()
