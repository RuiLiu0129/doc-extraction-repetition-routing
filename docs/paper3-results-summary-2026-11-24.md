# Paper 3 — Results Summary (measurement batches, Nov 22–23, 2026)

All numbers below were recomputed from the result files in `results/` on Nov 24, 2026
(micro P/R/F1; recall denominator = all gold fields in the split). Files:
`openrouter_*_test102.jsonl`, `openrouter_*_kleister-dev0.jsonl`,
`openrouter_*_extractbench-short.jsonl`, `openrouter_*_probe_*.jsonl`,
`cuad_train_repetition.jsonl`, `kleister_train_repetition.jsonl`.

## Method caveats (read first)

- Every run truncates documents to `max_chars=12000`; the median CUAD doc is ~25k
  chars, so absolute F1 is depressed by truncation, not only by model quality.
- Scoring is exact match, which punishes format differences (dates especially).
- Costs are OpenRouter list prices as reported Nov 22–23. Total spend $45.52 of the
  $50 top-up; **$4.48 remains**.

## RQ1 — Headline F1 by model and dataset

| Model | CUAD test (102 docs, 605 gold) | Kleister dev-0 (440 docs, 3,423 gold) | ExtractBench short (252 docs, 7,383 gold) |
|---|---:|---:|---:|
| gpt-5-mini | 0.310 | 0.514 | 0.322 |
| qwen3-235b-a22b-2507 | 0.334 | 0.506 | 0.307 |
| gemini-2.5-flash | 0.342 | 0.491 | 0.284 |
| claude-haiku-4.5 | 0.353 | 0.548 | 0.348 |
| claude-sonnet-4.5 | 0.370 | 0.547 | — (not run) |
| gpt-5 | 0.375 | 0.556 | — (not run) |
| gemini-2.5-pro | 0.381 | 0.563 | — (not run) |

Cheap-vs-expensive gap nearly vanishes on Kleister (haiku 0.548 vs gemini-2.5-pro
0.563, gap 0.015). Qwen costs ~$0.00057/doc, ~42x cheaper than GPT-5 for 0.05
less F1 on Kleister.

## Straight-through processing (STP): 0 everywhere

A document counts as STP only if every gold field is found and correct.
Verified against gold counts per doc: **CUAD 0/102, Kleister 0/440, ExtractBench
0/252, probe arms 0/50** — for every model. No model ever produces a fully
correct document.

## Repetition strata (train shares)

| Dataset | single | repeated-identical | repeated-conflicting |
|---|---:|---:|---:|
| CUAD train (2,598 fields) | 56.5% | 18.3% | 25.2% |
| Kleister train (13,449 fields) | 53.0% | 40.3% | 6.7% |

CUAD conflicting is concentrated in Parties (407/407 conflicting); non-Parties
conflicting is 9.5%.

### How conflicting breaks models — and it differs by dataset

**CUAD test102** (attempted accuracy by stratum): conflicting 0.01–0.05 for every
model (0.06–0.19 excluding Parties); identical 0.88–0.94 except mini 0.70;
single 0.34–0.49. Failure mode: **wrong-mention picks** — the model answers,
with the wrong value.

**Kleister dev-0**: conflicting attempted accuracy 0.59–0.82, but only ~40% of
the 234 conflicting gold fields are even attempted (88–111 per model). Failure
mode: **omission**, not wrong picks. Conflicting recall is 0.26–0.34.

## RQ2 — Injected-conflict probe v2 (50 CUAD test docs per arm)

Target-field accuracy (date field; control = same docs+field in Batch 1):

| Model | control | identical arm | conflicting arm | picked injected value |
|---|---:|---:|---:|---:|
| gpt-5-mini | 18/50 | 16/50 | 12/50 | 0/50 |
| claude-haiku-4.5 | 34/50 | 33/50 | 32/50 | 0/50 |
| gemini-2.5-flash | 26/50 | 28/50 | 20/50 | 6/50 |
| qwen3-235b-a22b-2507 | 35/50 | 33/50 | 15/50 | 14/50 |

Identical repetition ≈ control for every model: repetition alone does not hurt.
Under conflict the failure mode is model-specific — qwen collapses and picks
the injected value (14/50); haiku is nearly immune (34→32, 0 picks); mini
fails differently (18→12, 0 picks). Probe v1 was discarded (injections fell
outside the 12k truncation window).

Calibration note (5-doc smoke): mini's confidence barely separates right from
wrong (mean 95.9 correct vs 90.7 wrong; 11 wrong answers scored ≥90). GPT-5
separates better (97.1 vs 84.9).

## ExtractBench caveats

- **Text-coverage-capped**: 64.8% of gold fields never occur in the extracted
  text (42 docs scanned); nested fields are 100% unwinnable under exact match.
- **Mixed OCR text**: mini and qwen were restarted mid-run and resumed on
  OCR-improved text for docs 119–251, so the full-split numbers mix text
  versions. Apples-to-apples subset (docs 0–118, same old text for all models,
  micro-F1 with the same gold-counting as the full split): haiku **0.309**,
  flash **0.275**, mini **0.260**, qwen **0.232**.
- Correction note: the subset line reported in chat on Nov 23
  (0.310 / 0.279 / 0.260 / 0.259) mixed recall denominators across models and
  could not be reproduced from the final files. Recomputed consistently, the
  values are 0.309 / 0.275 / 0.260 / 0.232. The ranking is unchanged.

## Known artifacts (do not mistake for model effects)

- **Kleister `report_date`: 0 correct for all 7 models** (432–442 attempted
  each). ISO gold dates never match UK text dates — 12.9% of Kleister gold is
  unwinnable; max achievable recall is 87.1%.
- **Sonnet key bug**: 151 predictions used single-underscore address keys
  (`address_post_town` vs gold double underscore), 0 correct; ~0.03 precision lost.
- **Kleister income exact-match recall capped ~70–78%** by OCR variants.
- Kleister has **no LICENCE file** — verify before publication.

## Spend and what remains

- OpenRouter: $45.52 used of $50; **$4.48 left**. No top-up decided yet.
- Remaining: ExtractBench expensive tier (est. ~$18–35, needs a top-up decision),
  a clean cheap-tier re-run on uniform OCR text, fine-tuning, first draft before
  end of 2026.
