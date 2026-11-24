# Document extraction: cost vs accuracy — related work scan v2

2026-11-21 · Supersedes v1 (2026-11-02) · Multi-page focus only · Updated 2026-11-22: three adjacent papers added from the Nov 22 sweep (ConfBench, Scout, copy-heavy); the gap still stands. Updated 2026-11-23: Multi-Agent Memory Inconsistency added (multi-agent conflict handoff); the gap still stands.

## Conclusion first

- The closest work is **Rajendran (2026, arXiv:2608.06607)** on pre-inference routing — verified live 2026-11-22. It owns the 1-page receipt/invoice routing story, and it shows routers **do not transfer across datasets** (CORD → SROIE AUC ~0.55). That is the null our post-confidence cascade must beat, on multi-page data.
- The closest cost–accuracy benchmark is **Kulkarni & Kulkarni (2026, arXiv:2603.22651)**, 4 multi-agent architectures on 10,000 SEC filings — but its labels are not public, so it cannot be reproduced. Our slot is the same comparison on **public, multi-page data, by repetition stratum**.
- No prior work, as far as the Nov 2026 scout found, tests routing **by repeated/conflicting mention stratum** on long documents. That is the gap.
- Nov 22 sweep: ConfBench (confidence calibration benchmark), Scout (cross-document similarity), and the copy-heavy OCR-LLM paper are all adjacent — none tests routing by repetition stratum, so the gap stands.
- Nov 23: Multi-Agent Memory Inconsistency (Zhang et al., arXiv:2611.22245) is adjacent, not a competitor — multi-agent delegation splits conflicting claims across workers' contexts and conflict detection falls (16.2% → 13.2%); it supports the single-model full-context setup, since detection is higher when one worker reads both claims.

## Papers

| Paper | Data (public?) | What it compared | Cost reported? | Overlap |
|---|---|---|---|---|
| **Pre-Inference Routing for Cost-Efficient Document Field Extraction** (Rajendran, 2026, arXiv:2608.06607) | CORD, SROIE, DeepForm, DocILE, POIE, VRDU (public, mostly 1-page) | Pre-inference router (13 doc features) chooses Haiku vs Opus before extraction | Yes: cost down 31–33% receipts, 77% degraded ad-buy, within 0.02 F1 of always-large — only when cheap fails often enough *and* failures are predictable | **High, but on 1-page data.** Our test is multi-page + post-confidence, not pre-routing |
| **Benchmarking Multi-Agent LLM Architectures for Financial Document Processing** (Kulkarni & Kulkarni, 2026, arXiv:2603.22651) | 10,000 SEC filings, 25 fields (labels **not public**) | 4 architectures: sequential / parallel fan-out / hierarchical supervisor-worker / reflexive self-correcting; 5 models | Yes: reflexive F1 0.943 at 2.3× cost; hierarchical F1 0.921 at 1.4× cost; routing + retry ablations | **High.** Closest cost paper, but not reproducible, no cascade-by-confidence, no error-type analysis |
| **Calibrated Confidence for Straight-Through Processing** (OCBC, 2026, arXiv:2609.20110) | DocILE, FATURA, VRDU (public) | Field-level confidence + conformal thresholds, two-model cross-check | No dollars; reports fraction auto-approved | Medium. Strong baseline for the confidence part — "when to hand to a human", not cost–accuracy comparison |
| **EXTRACTCONF: Beyond Logprobs** (Perfios, 2026, arXiv:2606.24420) | DocILE, CORD (public) | Two calls in different ways + OCR features fused for confidence; isotonic calibration | Call count only (2 vs 5 for self-consistency) | Medium. Confidence method to compare against; single model (GPT-4o) only |
| **OCR or Not?** (SAP + Stanford, 2026, arXiv:2603.02789) | ~1,000 internal commercial docs (not public) | Image-only / OCR-text-only / image+OCR; ~10 multimodal models; automatic error-type analysis | Per-page cost + latency estimates | Medium. Input-mode comparison, no self-check or multi-agent. Authors note they did not try self-reflection |
| **Agentic System with RL Subsystem Improvements for Form-Like Documents** (Amjad et al., AAMAS-EMAS 2025, arXiv:2505.13504) | CORD, SROIE (public) + large private data | Single extraction vs multi-agent + RL prompt tuning; GPT-4o-mini | Time only (≥1 min vs <10 s), no dollars | Low–medium. 1-page data; shows agents help on CORD/SROIE (F1 0.886→0.965) but no cost |
| **MADP** (2026, arXiv:2605.17159) | 955 production docs (not public) | 5-agent pipeline + human review | Human + carbon, no dollars | Low |
| **Double Triangle Annotation** (EPFL, ACL 2026 Findings / arXiv:2605.25781) | Historical docs (benchmark public) | Two models cross-check, disagreements to human | Human-review fraction only | Low–medium. "Two models agree → auto-accept" is a usable baseline |
| **LongExtractionBench** (micro1/Reducto, 2026) | 225 docs, avg 358 pp, ~88,700 GT fields/doc — **full corpus not publicly released** | Dedicated platforms vs frontier LLMs on long dense docs; recall 49%–99.6% is the separator; frontier LLMs completed only 112–198/225 | No | Motivation / citation only. Do not plan experiments on it |
| **Can You Trust the Confidence? ConfBench for Vision-Language Models on Document Extraction** (2026, arXiv:2608.01792) | Degraded variants of diverse docs, 1,346 variants / 70K+ entity evals (public) | Verbalized vs logprob confidence, calibration quality across 4 proprietary + 3 open VLMs, routing automation vs human review | Review-budget metric ECARB, no dollars | **Medium.** Confidence calibration benchmark — validates confidence-based routing as a live concern and gives calibration methods for RQ2; but routes to humans, not to a bigger model, and is not repetition-stratified |
| **Scout: Scalable Document Extraction via Data Similarity** (2026, arXiv:2608.08261) | 6 real-world collections (public) | Rule generation from sampled docs, exploits answers recurring in similar locations across similar documents; cascade falls back to unrefined rules | Yes: 61×–1000× cheaper than a frontier LLM agent at matched accuracy | **Medium–low.** Similarity is *across* documents in a collection, not repetition/conflict *within* one long document; no cheap→expensive model routing |
| **Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task** (2025, arXiv:2510.10138) | Identity documents, 4 formats (dataset availability unclear) | Format-aware strategy selection across 25 configs (direct / replacement / table-based) + PaddleOCR | Latency only (sub-second), no dollars | **Low–medium.** "Copy-heavy" = many structurally similar documents, not a field value repeating/conflicting within one document; no model routing by confidence |
| **Multi-Agent Memory Inconsistency: Where Factual Conflicts Disappear in LLM Agents** (Zhang et al., 2026, arXiv:2611.22245) | Harvey LAB document-review tasks with controlled factual conflicts (benchmark availability not verified) | Multi-agent delegation vs single-agent: retrieval of both conflicting claims and conflict detection; worker-to-lead handoff profiling | Generation spending matched across settings, no dollars | **Low–medium.** Adjacent, not a competitor: conflicts split across agents' contexts and detection falls when the claims are split (16.2% → 13.2% with delegation); our setting is single-model extraction from one document, and its "one worker reads both claims detects more" finding supports the full-context setup |

## Datasets in this paper (locked 2026-11-21)

- **Kleister Charity** — 2,778 docs, 21,612 entities, 8 fields, mean 22.19 pp/doc. Baselines in paper: Flair/BERT/RoBERTa/LayoutLM/LAMBERT, best F1 83.57.
- **CUAD** — 510 contracts, 13,823 spans, 41 labels. RoBERTa/DeBERTa baselines.
- **ExtractBench** — 370 docs, 67 schemas, T1 long-list / T2 needle-in-haystack / T3. 14 systems incl. GPT-5.5, Claude Opus 4.8, Gemini 3 Flash. **No small fine-tuned transformer baseline — the gap this paper fills.**
- **Full Form 990** — second-domain test-only, ~200 docs, tax year 2023. No annotated dataset exists; value-only GT from IRS e-file XML.

## What is new here (ranked)

1. **Repetition-stratum comparison:** same spend, but accuracy reported by single / repeated-identical / repeated-conflicting occurrence — not just an overall F1.
2. **Post-confidence cascade vs pre-routing, at matched spend, on multi-page data** — including the cross-dataset transfer test Rajendran shows fails for pre-routing.
3. **All public, reproducible:** Kleister + CUAD + ExtractBench (+ 990 value-only), with all model outputs released.
4. **Error-type analysis by stratum:** omission vs wrong-occurrence vs hallucination, and which confidence signal catches each.
5. **Multi-vendor models:** cheap and expensive tiers from OpenAI, Google, Anthropic, plus open models — most prior work uses 1–2 models.

## Risks

- Top-venue reviewers may call this engineering, not a new method. The defence is the repetition-stratum finding + the calibrated cascade, on public data, with a fair same-spend comparison. Best fit is a workshop / industry track, per the Oct 23 strategy.
- Rajendran must be cited head-on, with its non-transfer result stated, not softened.
- Budget: multi-page docs cost more per doc in tokens than 1-page receipts. Re-estimate in Week 1 from measured token counts; the $500 cap is hard.
