# Paper 3: Document extraction — repetition-first model comparison and confidence-gated routing (design v2)

2026-11-21 · Supersedes the other-AI design v1 (2026-11-02) · Public data only · Budget hard cap $500

Author block (per 2026-10-22 decision): Rui Liu / Independent Researcher / RL2987@columbia.edu — no Meta affiliation.

## 0. What changed from v1 — read this first

v1 was written before the dataset scout (2026-11-21/22). Five corrections, all from that discussion:

1. **Datasets are locked and multi-page only.** v1 used CORD, SROIE and DocILE as primary. Those are 1-page receipts/invoices, already saturated (LayoutLMv2 F1 0.96 on CORD, 0.978 on SROIE), and the user explicitly said: one-page datasets, don't look at them, focus on multi-page. They are **dropped as primary** — not even as a sanity check in the main results.
   Locked: **Kleister Charity + CUAD + ExtractBench (test-only)**, plus **full Form 990 as a second-domain, test-only set (~200 docs)**. See §4.
2. **The research question is repetition-first, not cascade-generic.** v1's selling point was "cheap first, escalate on confidence" in general. The locked question is narrower and is the user's own pain point from 5 years at AYR/Singularity: **when the same field value appears multiple times / with conflicting values in a long complex document, which model gets it right, and can a cheap signal predict which model to use?** Cascade vs pre-routing is still the mechanism comparison (§8), but it is tested *by repetition stratum*, not on average accuracy alone.
3. **step1-setup.md in v1 is deleted, not edited.** It told the user to run τ²-bench (airline agent benchmark) and SWE-bench (coding, Docker/x86). That is a different project entirely — wrong benchmark, wrong task, wrong setup. It is kept in `docs/` as `step1-setup-v1-OTHER-AI-ORIGINAL.md` for audit only. The replacement is `step1-setup-v2.md` in this repo.
4. **step1_cord_smoke.py is deleted, not edited.** It was a CORD-only (1-page receipt) smoke test. Replacement: `src/step1_multipage_smoke.py` — CUAD text smoke + Kleister loader + repetition-stratum measurement, with a no-API dry-run mode.
5. **ICDAR 2027 dates in v1 are not reliable.** v1 said abstract Jan 31, 2027 / paper Feb 20, 2027. Live deadline listings on 2026-11-22 disagree (Feb 20 / Feb 28 / Mar 5, 2027, depending on source). **Do not cite a date in the paper or plan around it until it is verified on icdar2027.org.** The user's hard goal is a first draft by Dec 31, 2026; the venue is decided after the draft, with workshop / industry track ("快中、多中") as the default per the Oct 23 strategy, not ICDAR main track by default.

## 1. One-sentence selling point

On long multi-page documents, cheap models fail first where a field value repeats — they pick the wrong occurrence, or miss it entirely — and a calibrated per-field confidence, plus cheap pre-features (occurrence count, document length, field type), predicts those failures well enough that a post-confidence cascade beats both "always cheap" and "always expensive" at the same spend, and beats pre-inference routing on the repeated/conflicting strata.

## 2. Closest work to answer head-on

**Pre-Inference Routing for Cost-Efficient Document Field Extraction** (Rajendran, 2026, arXiv:2608.06607, submitted Aug 6, 2026 — verified live 2026-11-22).
- Pre-inference router: 13 document features (OCR quality, layout, etc.) predict difficulty *before* calling a model, then choose Haiku vs Opus.
- Data: CORD, SROIE, DeepForm, DocILE, POIE, VRDU — almost exactly v1's dataset list, which is part of why v1's list is now dropped: that paper already owns the 1-page receipt/invoice routing story.
- Results (from its abstract): cost down 31–33% on receipts and 77% on degraded ad-buy forms, quality within 0.02 F1 of always-large — **but only when two conditions hold** (cheap model fails often enough, and failures are predictable from visible features), and **the router does not transfer across datasets, even within the same genre** (CORD → SROIE AUC ~0.55).
- **Our difference:** (1) escalation uses *post-output* per-field confidence (self-reported confidence, logprob where available, multi-sample agreement, two-cheap-model agreement, OCR/text-match), calibrated with Platt / isotonic — Rajendran notes Anthropic gives no logprob, so this was not done there; (2) multi-vendor models, not Anthropic only; (3) the test is on **long multi-page documents with measured repetition strata**, not 1-page receipts; (4) direct same-spend comparison of pre-routing vs post-confidence cascade, *by stratum*.

**Unverified claim in v1 — do not repeat it:** v1 said Rajendran "also compared cheap-first cascade and found it worse than pre-routing". That comparison is **not in the abstract** and was not verified in the full text on 2026-11-22. Do not cite it as a finding. If the full text does contain it, cite the section number after reading it; until then, treat it as unconfirmed.

Other related work is in `docs/doc-extraction-related-work-v2.md`. The other multi-agent cost paper (Kulkarni & Kulkarni, arXiv:2603.22651, SEC filings, F1 0.943 reflexive at 2.3× cost) is the closest cost–accuracy benchmark, but its data are not public — our contribution is the same comparison on public data, by repetition stratum.

## 3. Research questions

- **RQ1 — Where do cheap models lose?** Field F1 for cheap vs expensive models, split by repetition stratum: single-occurrence / repeated-identical / repeated-conflicting (see §5 for definitions). Hypothesis: the gap is small on single-occurrence fields and large on repeated-conflicting fields.
- **RQ2 — What predicts the failure?** Which signal best detects a cheap model's error: self-reported confidence, logprob, multi-sample agreement, two-model agreement, OCR/text-match, or cheap pre-features (occurrence count, doc length in pages/tokens, field type, position of first occurrence)? Report AUROC, ECE after Platt/isotonic calibration, and risk–coverage (AURC). Test calibration transfer: fit on Kleister, apply to CUAD, and vice versa.
- **RQ3 — Same spend, which routing wins?** At matched total spend, compare: always-cheap / always-expensive / pre-routing (simplified Rajendran replication: document features + random forest) / **post-confidence cascade (ours)** / self-check / two-cheap cross-check / oracle (knows which docs are wrong, then escalates — theoretical upper bound). Draw cost–accuracy Pareto curves, overall and by stratum.
- **RQ4 — Plus a human layer (optional, only if RQ1–RQ3 are done).** Field-level human review for low-confidence fields. Total cost = API cost + reviewed fields × human unit price, scanned over unit prices. If time is short in December, RQ4 becomes a 1-paragraph sensitivity note, not a full experiment.

## 4. Data — locked, all public, multi-page only

| Dataset | Domain | Size | Length | Split to use | Ground truth | Source |
|---|---|---|---|---|---|---|
| **Kleister Charity** (primary) | UK charity annual financial reports | 2,778 docs*; 21,612 entities; 8 fields | Mean 22.19 pp / 5,149 words per doc | Train 1,729 / dev 440 / test 609. Paper: fine-tune small models on a 300-doc subsample of train; validate routing on dev (200); report on test (200, stratified by repetition) | Value-only JSON (no span locations) + PDF + OCR text | GitHub `applicaai/kleister-charity` — *count discrepancy: paper/table say 2,778 vs 2,788; verify on download. Verify repo LICENCE before publication; test annotations may not be shipped — if absent, use dev 440 as held-out test* |
| **CUAD** (second) | Commercial contracts (EDGAR) | 510 docs; 13,823 spans; 41 labels | Median ~33k chars / ~8k tokens (secondary source, indicative) | Train 408 / test 102 contracts. Fine-tune on 200 of train; test on all 102. Use ~10 of the 41 labels that are short values (dates, parties, amounts), state this explicitly — long-clause extraction is a different task | SQuAD-style JSON: clause → [{text, char start/end}] + full text + PDF | HF `theatticusproject/cuad`, CC BY 4.0 |
| **ExtractBench** (test-only) | Enterprise: finance, energy, govt, auto, supply-chain, healthcare remittance, legal/bankruptcy, real estate | 370 docs; 841,934 expected values; 67 schemas | Short ≤10 pp (252), medium 11–50 (98), long >50 (20); 4,869 pp total | **No train split — test only.** Short (252) as main held-out test (or stratified 100-doc subsample); **T2 needle-in-haystack (39 docs) as the repetition stress test**; long (20) as extreme-length probe | JSON schema + expected JSON + per-field evidence {page, bbox, quote} for 339/370; PDF + bbox | HF `llamaindex/ExtractBench` — *licence not stated on HF card as read 2026-11-22; verify in repo before redistributing* |
| **Full Form 990** (second-domain, test-only, ~200 docs) | US nonprofit tax filings | Build 200 docs, tax year 2023 only | Core 12 pp + schedules; real small filing 33 pp; 200 docs ≈ 4,000–7,000+ pp | Test only, single tax year 2023, schema TY2023v6.0, one filing per EIN, ~50% with Schedule J | Value-only, from XML: Dalton HF `mauricedalton/irs-990` (6.2M filings; 3.97M full 990 + 2.24M EZ; CC0) joined to ProPublica `pdf_url` for the PDF. 10–12 short fields (§5) | No annotated 990 dataset exists (searched HF/GitHub/arXiv 2026-11-22). Do not claim span-level evaluation on 990. Report rendered-PDF vs scanned status separately |

*Excluded, with reason (do not re-add without a new decision):* FUNSD (199 docs, 1 pp, saturated), CORD (1,000 docs, 1 pp, LayoutLMv2 F1 0.96), SROIE (973 docs, 1 pp, F1 0.978), DocILE as primary (1–3 pp, hidden test labels, token-gated download), FinanceBench (only 150 open questions), LexGLUE (classification, not extraction), LongExtractionBench (full 225-doc corpus not publicly released — cite as motivation only), BuDDIE (non-commercial, on request).

## 5. Repetition strata — the core measurement (Week 1, no new labelling)

For each gold field value in a document, normalise it (numbers/dates/names: strip $, commas, whitespace; parentheses = negative) and count exact occurrences in the document text, recording position (page/section), context (table vs narrative vs header/footer), and whether occurrences agree.

- **Single:** 1 occurrence.
- **Repeated-identical:** ≥2 occurrences, all with the same normalised value.
- **Repeated-conflicting:** ≥2 occurrences with different normalised values (e.g. prior-year restatement, amended figure, "total" vs "subtotal", Part VII vs Schedule J compensation).

Report stratum sizes per dataset. If the conflicting stratum is <30 docs, oversample it for the test set. This measurement is part of the method, not a pre-known fact — for every dataset except ExtractBench T1/T2, repetition prevalence is inferred, not labelled.

**Small injected-conflict probe (the paper's novel artefact, 50–100 docs, 2–4 days, no human labelling):** take test docs from Kleister/CUAD and inject one controlled perturbation per doc, logged in a JSON manifest: (i) duplicate a paragraph/table row containing the gold value verbatim elsewhere (identical-repeat); (ii) duplicate it with the value changed and a plausible cue ("as restated", "prior year", "amended") — gold stays the original canonical occurrence; (iii) move the gold value to a late page / bury it among distractors (position probe). Keep the original as control. Paired design: same doc, same gold, only repetition changes — any accuracy drop is attributable to repetition. Release the manifest + script, not the modified docs alone. Validate the routing rule learned on Kleister/CUAD on ExtractBench T2 (unmodified, 39 docs).

**Full 990 fields (10–12, short values only):** org name, EIN, tax period end, total revenue (current year), total expenses (current year), contributions & grants, program service revenue, total assets EOY, total liabilities EOY, net assets EOY, total fundraising expenses, optional top-officer reportable compensation (hardest — Part VII vs Schedule J trap; drop it if time is short and use employee count instead). Gold definition is the canonical occurrence (e.g. Part I current-year column); prior-year / beginning-of-year values score as wrong.

## 6. Models (all through one OpenRouter key)

| Tier | Models (verified on openrouter.ai/models, 2026-11-22) |
|---|---|
| Cheap | GPT-5-mini (`openai/gpt-5-mini`), Gemini Flash (`google/gemini-2.5-flash`), Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`), Qwen (`qwen/qwen3-235b-a22b-2507`, open — the undated `qwen3-235b-a22b` id 404s as of 2026-11-22; `qwen/qwen3-vl-235b-a22b-instruct` is the VL alternative) |
| Expensive | GPT-5 (`openai/gpt-5`), Gemini Pro (`google/gemini-2.5-pro`), Claude Sonnet (`anthropic/claude-sonnet-4.5`; 4.6/5/5.5 also listed) |

Fine-tuned small baseline (trainable only on Kleister + CUAD, which have spans/values to train on): LayoutLMv3 / BERT / RoBERTa, fine-tuned on the 300 + 200 train subsamples in §4. ExtractBench and 990 have no train split — they are test-only for all models, which is exactly the gap to report: no small fine-tuned baseline exists there yet.

## 7. Core cost trick: run each model once per document, simulate everything offline

- Store every model's output + spend per document.
- All cascade thresholds and cheap→expensive combinations are computed offline from the stored outputs. **No repeat API spend.**
- 4 cheap × 3 expensive = 12 combinations × any threshold, free after the first pass.

## 8. Confidence signals (cheap model)

| Signal | Extra cost |
|---|---|
| Self-reported confidence per field (0–100) | 0 |
| Logprob (OpenAI / open models where available) | 0 |
| OCR/text match: is the extracted value findable in the document text? | 0 (local) |
| Multi-sample agreement (same model, 2 extra runs) | +2 cheap calls |
| Two cheap models agree? | Uses stored outputs, no extra cost |
| Self-check: show the model its own result, ask "is anything wrong?" | +1 cheap call |

Calibration: Platt, isotonic, fit on train, evaluate on test; plus cross-dataset transfer (fit on Kleister, apply to CUAD, and vice versa). Rajendran's non-transfer result is the null to beat — report it even if our calibration also fails to transfer.

## 9. Comparison methods

1. Always cheap
2. Always expensive (accuracy ceiling reference)
3. Pre-routing (simplified Rajendran replication: document features + random forest, fit on train)
4. **Post-confidence cascade (ours):** if the lowest field confidence in a document is below threshold, escalate the whole document to the expensive model, then merge field-level results
5. Self-check (cheap model checks itself once)
6. Two cheap models cross-check; escalate on disagreement
7. Oracle: escalate only docs known to be wrong (theoretical upper bound)

## 10. Metrics

- Field-level F1 (normalised exact match), document-level exact-match rate — **reported by repetition stratum, not just overall**
- Straight-through processing (STP) rate: fraction of documents where every gold field is found and correct, i.e. STP = 1 for a document only when correct count equals gold count for that document, with no missing fields. This is the automation rate: an STP = 1 document needs no human touch. Report per model, and by the repetition stratum of the document's fields. Note: on the 5-doc smoke runs (Nov 22) both gpt-5-mini and GPT-5 scored STP 0/5, which is expected at per-field accuracy near 0.4 with 3–8 fields per doc.
- Cost per 1,000 documents, mean latency
- Cost–accuracy curve and Pareto frontier; "spend needed to reach 99% of expensive-model F1"
- Confidence: AUROC, ECE, risk–coverage curve (AURC)
- Total cost = API cost + human-reviewed fields × human unit price (scan unit prices; RQ4 only)
- Error types: wrong character / wrong field / wrong format / hallucination / omission — by stratum
- Statistics: per-document bootstrap confidence intervals

## 11. Budget (rough, OpenRouter list prices — re-check at run time)

| Item | Scale | Estimate |
|---|---|---|
| Cheap models, test + train docs | 4 models × ~1,850 docs × ~4 calls (1 main + 2 sample + 1 self-check) — **recompute for multi-page: Kleister/CUAD test sets are ~300 docs, not 1,850; the 1,850 figure in v1 was CORD/SROIE/DocILE scale and no longer applies** | Re-estimate in Week 1 from actual page/token counts; keep the $500 hard cap |
| Expensive models, test docs | 3 models × ~400 docs × 1 call (Kleister test 200 + CUAD test 102 + ExtractBench ~100) | Re-estimate in Week 1 |
| Trial + rerun buffer | | $30–50 |
| **Total** | | **Hard cap $500. Start with $50 on OpenRouter, set a credit limit, top up only if the Week 1 estimate fits the cap** |

No GPU needed. Local analysis runs on a laptop. Multi-page PDFs are the cost driver (tokens per doc scale with pages) — this is why the Week 1 token-count measurement matters before any large run.

## 12. Timeline (user goal: first draft by Dec 31, 2026)

| When | Task |
|---|---|
| Nov 21–26, 2026 | Dataset lock (done) + this v2 design. Paper 2 camera-ready (Nov 26) takes priority — no large API runs this week |
| Dec 1 | Paper 2 video due. USVI trip Nov 29–Dec 5 |
| Dec 6–12 | Week 1: load Kleister + CUAD, measure repetition strata, build injected-conflict probe, freeze prompts/schemas, measure tokens/cost, re-estimate budget |
| Dec 13–19 | Week 2: fine-tune small models (300 Kleister + 200 CUAD train docs) vs few-shot LLM vs large LLM on the same test sets; log per-field occurrence count, position, doc length, layout features |
| Dec 20–31 | Week 3: fit routing rule on dev, validate on ExtractBench T2/short + 990 (200 docs), write first draft |
| Jan 2027 | Confidence, calibration, cascade offline simulation, pre-routing baseline, error analysis, draft revision |
| Venue | **Decide after the draft.** ICDAR 2027 is a candidate only — verify dates on icdar2027.org (listings disagreed in Nov 2026). Default per Oct 23 strategy: mid-tier, higher-acceptance, fast-cycle workshop / industry track ("快中、多中"), arXiv in parallel |

## 13. Risks

- Rajendran (2026) already owns 1-page routing and shows routers do not transfer across datasets. Our claim must survive the same test on multi-page data — report cross-dataset transfer honestly, even if it fails.
- Public datasets may be in frontier models' training data. This inflates both cheap and expensive scores; the comparison (gap + spend) is less affected. State it in the paper.
- ExtractBench is very new (Jul–Sep 2026) and already saturated by vendor systems at 95–99% on some splits — use it as a stress test (T2), not as the headline number.
- 990 is a fixed template: a small model can memorise line positions. Do not over-claim generalisation from 990; frame it as a templated-form second domain.
- Kleister test annotations may not be shipped — fallback is dev 440 as held-out test (see §4).
- DocILE token / any new dataset access friction: not needed anymore — DocILE is dropped as primary.

## 14. Step 1 (about 1 hour, <$1)

See `step1-setup-v2.md`. In short: OpenRouter key, install, run `src/check_keys.py`, run `src/step1_multipage_smoke.py --dry-run`, then `--run-llm --dataset cuad --n 5`.
