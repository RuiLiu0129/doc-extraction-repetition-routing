# Paper 3: Document extraction — repetition-first routing

Comparing cheap vs expensive models on **long multi-page documents**,
especially where a field value **repeats** — identical or conflicting —
and learning when to use which model.

Author: Rui Liu / Independent Researcher / RL2987@columbia.edu (no Meta affiliation).
First draft goal: Dec 31, 2026.

## Locked datasets (2026-11-21, multi-page only)

| Dataset | Role | Size |
|---|---|---|
| Kleister Charity | Primary | 2,778 docs, 8 fields, mean 22.19 pp/doc |
| CUAD | Second | 510 contracts, 41 labels (use ~10 short-value labels) |
| ExtractBench | Test-only | 370 docs, T2 needle-in-haystack (39 docs) is the repetition stress test |
| Full Form 990 | Second-domain, test-only (~200 docs, tax year 2023) | Value-only GT from IRS e-file XML; no annotated dataset exists |

One-page datasets (CORD, SROIE, FUNSD, DocILE as primary) are **excluded**:
saturated and the wrong shape for this paper. See `doc-extraction-design-v2.md` §4.

## What is in this repo

- `doc-extraction-design-v2.md` — experiment design v2 (supersedes the other-AI v1)
- `docs/doc-extraction-related-work-v2.md` — related-work scan v2
- `docs/*-OTHER-AI-ORIGINAL.md` — the other AI's v1 files, kept for audit only. **Do not follow them.** In particular, `step1-setup-v1-OTHER-AI-ORIGINAL.md` tells you to run τ²-bench / SWE-bench — a different project.
- `step1-setup-v2.md` — Step 1: OpenRouter key, dataset downloads, smoke test
- `src/check_keys.py` — check the OpenRouter key against 5 models
- `src/step1_multipage_smoke.py` — multi-page smoke test (CUAD / Kleister / ExtractBench), dry-run by default
- `src/measure_repetition.py` — Week 1 repetition-stratum measurement (CUAD now; Kleister after layout check)

## Quick start

```bash
pip install -r requirements.txt
# .env with OPENROUTER_API_KEY=... (never commit it)
python src/check_keys.py
python src/step1_multipage_smoke.py --dry-run --dataset cuad --n 5
python src/step1_multipage_smoke.py --run-llm --dataset cuad --n 5
```

## Budget

Hard cap **$500** total API spend. Start with $50 on OpenRouter and a
credit limit. Re-estimate in Week 1 from measured token counts —
multi-page docs cost more per doc than 1-page receipts.

## Status

- 2026-11-21: datasets locked (Kleister + CUAD + ExtractBench; 990 second-domain), design v2 written.
- Next: Week 1 (Dec 6–12) — repetition measurement, injected-conflict probe, token/cost estimate.
