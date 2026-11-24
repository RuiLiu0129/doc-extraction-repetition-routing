# Step 1: setup (v2) — about 1 hour, <$1

Replaces the v1 setup file entirely. The v1 file told you to run τ²-bench
(airline agent benchmark) and SWE-bench (coding benchmark, Docker/x86).
That was a different project. Do not run it for this paper.

## 1. One API key: OpenRouter

- Register at openrouter.ai, top up **$50 to start** (hard cap for the whole
  paper is $500 — top up only if the Week 1 cost estimate fits).
- Set a credit limit in OpenRouter settings, e.g. $50.
- One key calls GPT, Claude, Gemini, Qwen, DeepSeek — no separate signups.
- Use a personal email and personal card, not a company account.

## 2. Check the key (5 minutes)

In this repo folder, create a `.env` file with one line:

```
OPENROUTER_API_KEY=your-key-here
```

Then:

```bash
pip install -r requirements.txt
python src/check_keys.py
```

It asks 5 models one arithmetic question each, prints OK/FAIL and cost
(total well under $0.01). If a model name errors, check
openrouter.ai/models for the current name and update `src/check_keys.py`.

**Never commit `.env`.** It is in `.gitignore`. Never paste the key in chat.

## 3. Download the locked datasets (no API spend)

```bash
# CUAD (text, easiest first smoke test) — canonical JSON from the authors'
# GitHub repo, NOT Hugging Face: HF theatticusproject/cuad no longer ships a
# loading script or annotations (only 511 PDFs) as of 2026-11-22.
curl -L -o data/data.zip https://github.com/The-Atticus-Project/cuad/raw/main/data.zip
cd data && unzip -o data.zip train_separate_questions.json test.json CUADv1.json && cd ..
# train_separate_questions.json = 408 train docs, test.json = 102 test docs.
python -c "import json; print(len(json.load(open('data/train_separate_questions.json'))['data']))"

# Kleister Charity (PDF + OCR + value-only JSON)
git clone https://github.com/applicaai/kleister-charity.git data/kleister-charity
# Check the repo LICENCE file, and whether test annotations are shipped.
# If test labels are absent, use dev (440 docs) as the held-out test.

# ExtractBench (test-only, no train split)
python -c "from datasets import load_dataset; ds = load_dataset('llamaindex/ExtractBench', split='short'); print(len(ds))"
# Verify the licence in https://github.com/run-llama/ExtractBench before redistributing.
```

Full Form 990 is **not** downloaded in Step 1. It is a second-domain,
test-only set built in Week 1 from tax-year 2023 filings (see design v2 §4).

## 4. Smoke test — dry run first, no API spend

```bash
python src/step1_multipage_smoke.py --dry-run
python src/step1_multipage_smoke.py --dry-run --dataset cuad --n 5
```

This loads data, counts repetition strata (single / repeated-identical /
repeated-conflicting) for gold values, and prints page/token stats.
It calls **no** model.

Then, with the key in place, one cheap-model smoke run:

```bash
python src/step1_multipage_smoke.py --run-llm --dataset cuad --n 5
python src/step1_multipage_smoke.py --run-llm --dataset cuad --n 5 --model openrouter/anthropic/claude-haiku-4.5
```

Output goes to `results/<model>_multipage_smoke.jsonl` — one row per
predicted field, with key, prediction, gold, correct, confidence,
and the gold value's occurrence count in the document.

## 5. What to send back

- `src/check_keys.py` output
- `src/step1_multipage_smoke.py --dry-run` output
- the `results/` files from the 5-doc LLM smoke run

Next step after that: the full batch runner + scoring code, and the
Week 1 repetition measurement on Kleister Charity.
