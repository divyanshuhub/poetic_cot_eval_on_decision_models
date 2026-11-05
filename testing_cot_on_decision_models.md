# Testing CoT on Decision Models

## Overview

`bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py` evaluates local fine-tuned decision models (Laya, Kev, Clef) and the hosted JEV comparison model against a vLLM- or Gemini-generated CoT CSV. It tests how supplied standard, poisoned, and poetic reasoning affects decision outcomes across multiple reasoning benchmarks. It does not train models or generate reasoning.

### Supported Reasoning Datasets

The evaluation pipeline natively supports **four diverse reasoning datasets**:
1. **AI2 ARC** (`allenai/ai2_arc`): Multiple-choice science questions (`ARC-Easy`, `ARC-Challenge`).
2. **LogiQA** (`lucasmccabe/logiqa`): Critical reading comprehension and logical deductive/inductive inference. Context is preserved in the decision state.
3. **Typed Decisions** (`LocalLLaMA/typed-decisions`): Multi-question operational decision-making scenarios with rich environment states.
4. **ProofWriter** (`tasksource/proofwriter`): Formal first-order logic derivation tasks over theories and queries (`True`, `False`, `Unknown`).

The dataset, configuration, and split are **automatically detected** from the input CSV filename or headers when `--dataset`, `--dataset-config`, or `--split` are omitted.

| Condition | What the model sees |
|---|---|
| `question_only` | Question + choices + dataset state/context, without rationale |
| `standard_reasoning` | Question + choices + dataset state/context + factual CoT |
| `poisoned_reasoning` | Question + choices + dataset state/context + misleading distractor CoT |
| `poetic_reasoning` | Question + choices + dataset state/context + five-line poetic CoT |

The trailing `The final answer is ####` line is stripped from `standard_cot`. Blank rationale cells are recorded as skipped for that condition; question-only still runs.

**Comparison caveat:** JEV is a hosted model, not a locally fine-tuned checkpoint. Report that distinction when comparing it with Laya/Kev/Clef.

## How `run_cuda.sh` Works

1. Sources the project-root `.env`, if present.
2. Uses `ARC_DECISION_PYTHON` if set, otherwise the workspace venv at `GSM8K/.venv/bin/python`.
3. Sets CUDA defaults: `CUDA_VISIBLE_DEVICES=0`, `LAYA_DEVICE=cuda`, and CUDA library paths.
4. Adds `arc_decision_reasoning/src` to `PYTHONPATH`.
5. Executes the requested Python script with its arguments.

JEV uses remote HTTP requests and requires neither a local JEV server nor local model weights. The wrapper remains useful for loading `.env` and configuring Python.

## The Evaluation Script and Project Layout

Entry point: `scripts/evaluate_cot_decision_model.py`, a thin wrapper around `src/arc_decision/cot_evaluation.py`.

arc_decision_reasoning/
├── .env
├── run_cuda.sh
├── enable_cot_concurrency.py
├── scripts/
│   └── evaluate_cot_decision_model.py
└── src/
    └── arc_decision/
        ├── config.py
        ├── decision_client.py
        └── cot_evaluation.py

Run all commands below from `arc_decision_reasoning/`.

## CLI Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--input-csv` | Yes | — | CSV containing questions, choices, answers, and `standard_cot`, `poisoned_cot`, `poetic_cot`. |
| `--dataset` / `--dataset-name` | No | Auto-detected | `arc`, `logiqa`, `typed_decisions`, or `proofwriter`. |
| `--dataset-config` / `--dataset-type` | No | Auto-detected | Subset/config: `easy`/`challenge` (ARC), `all`/scenarios (Typed Decisions), `default` (LogiQA, ProofWriter). |
| `--split` | No | `test` / auto-detected | `train`, `validation`, or `test`. |
| `--decision-model` | Yes | — | `laya`, `kev`, `clef`, or `jev`. |
| `--output-dir` | No | `outputs/cot_evaluations/{stem}_{model}_finetuned` | Result directory inside the project. |
| `--laya-finetuned-checkpoint` | For Laya | — | Local fine-tuned Laya checkpoint directory; do not pass for JEV. |
| `--kev-finetuned-checkpoint` | No | `KEV_MODEL_ID` in `.env` | Override fine-tuned Kev checkpoint directory from CLI. |
| `--clef-finetuned-checkpoint` | No | `CLEF_MODEL_ID` in `.env` | Override fine-tuned Clef checkpoint directory from CLI. |
| `--limit` | No | All rows | Evaluate the first N rows; N must be positive. |
| `--max-retries` | No | `2` | Up to two retries after the initial failed request with exponential backoff. |
| `--request-timeout` | No | `60.0` | Positive per-request timeout in seconds. |
| `--concurrency` | No | `1` | Maximum simultaneous HTTP calls for Kev/Clef/JEV. Laya requires 1. |
| `--resume` | No | False | Skip matching completed work and retry unsuccessful work. |
| `--force-regenerate` | No | False | Re-evaluate conditions despite cached results. |

**Installation status:** `--concurrency` is available only after the patch succeeds. The earlier installer attempt failed because it targeted `arc_decision/cot_evaluation.py` rather than the documented `src/arc_decision/cot_evaluation.py`. That failed attempt changed no files.

## How Each Decision Model Is Connected

### Laya: In-Process, Local

Laya runs through the `laya` Python package and `laya.Router`, without a server.

- Install: `pip install -e '.[laya]'`.
- Supply `--laya-finetuned-checkpoint`.
- `LAYA_DEVICE=cuda` is set by the wrapper.
- Optional variables: `LAYA_PRELOAD=0` (use 1 to preload) and `LAYA_CHECKPOINT` (base route such as `typed-decisions`).

The checkpoint directory is recorded as the model identity and checkpoint path. Keep concurrency at 1.

### Kev: Local HTTP Server

Default endpoint: `http://127.0.0.1:8008/v1/systemone`.

1. Clone Kev if needed: `git clone https://github.com/jaredpalmer/kev.git ./kev`.
2. Set `KEV_MODEL_ID` to the fine-tuned checkpoint.
3. Start `bash ./run_kev_server.sh` separately before evaluation.

| Variable | Default / purpose |
|---|---|
| `KEV_MODEL_ID` | Base default `jaredpalmer/kev-4b`; the evaluator rejects that base model, so override with a fine-tuned checkpoint. |
| `KEV_API_URL` | `http://127.0.0.1:8008/v1/systemone` |
| `KEV_API_KEY` | Optional authentication for the local server. |
| `KEV_DTYPE` | `bf16` |
| `KEV_REPO_DIR` | Auto-detected from `./kev`; override for your installation. |

### Clef: Local HTTP Server via SGLang Container

Start `bash ./run_clef_server.sh` before evaluation.

| Variable | Default |
|---|---|
| `CLEF_MODEL_ID` | `Cloudflare/clef-flash` |
| `CLEF_API_URL` | `http://127.0.0.1:30000/v1/systemone` |

### JEV: Hosted TypeSafe API

JEV uses `POST https://api.typesafe.ai/v1/systemone` with bearer authentication. The integration reuses the SystemOne HTTP adapter to send `state`, `model`, and typed `questions`, and parse structured answers.

Add credentials to the **same project-root `.env` used by Kev**:

```bash
JEV_API_KEY=your_actual_api_key
JEV_MODEL_ID=jev-latest
JEV_API_URL=https://api.typesafe.ai/v1/systemone
```

| Variable | Default / purpose |
|---|---|
| `JEV_API_KEY` | Required unless `TYPESAFE_API_KEY` is set; preferred when both are present. |
| `TYPESAFE_API_KEY` | Alternative key name accepted by the integration. |
| `JEV_MODEL_ID` | `jev-latest` |
| `JEV_API_URL` | `https://api.typesafe.ai/v1/systemone`; the integration requires HTTPS. |

Do not share or commit API keys. Do not pass a Laya checkpoint for JEV or start a JEV server.

Use `jev-latest` for the smoke test. For final experiments, inspect the returned `model` field and pin that exact version if the API accepts it. Use one fixed version across all conditions and repeat runs. Do not assume the documentation example `jev-1.13.0` is the current model.

Documentation: https://docs.typesafe.ai/introduction and https://docs.typesafe.ai/introduction/quickstart.

### JEV Integration Changes

- `src/arc_decision/config.py`: add `jev` to `DECISION_MODELS`.
- `src/arc_decision/decision_client.py`: resolve JEV environment settings with `jev_systemone_config`; route JEV through the HTTP client factory.
- `src/arc_decision/cot_evaluation.py`: resolve JEV model identity separately, without requiring a local fine-tuned checkpoint.
- HTTP errors identify the selected model and API rather than always referring to a local server.
- The thin launcher remains unchanged.

## Complete Test Cases

### Test Case 1: Laya Fine-Tuned on ARC-Easy

The checkpoint and input CSV must exist.

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv \
  --dataset-config easy --split test \
  --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_arc_easy/checkpoint \
  --output-dir outputs/cot_evaluations/arc_easy_laya_ft
```

Loads and normalizes the CSV, registers the fine-tuned checkpoint as `arc_finetuned` in the router, evaluates available conditions, and writes the output artifacts. Add `--resume` to continue an interrupted run.

### Test Case 2: Kev Fine-Tuned on ARC-Easy

Keep your existing Kev configuration in `.env`:

KEV_REPO_DIR=/home/gpuuser7/gpuuser7_a/Divyanshu/GSM8K/arc_decision_reasoning/kev/kev
KEV_MODEL_ID=/home/gpuuser7/gpuuser7_a/Divyanshu/GSM8K/arc_decision_reasoning/outputs/finetunes/kev_arc_easy/checkpoint

Start the server in a separate terminal:

```bash
bash ./run_kev_server.sh
```

Then evaluate:

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv \
  --dataset-config easy --split test \
  --decision-model kev \
  --output-dir outputs/cot_evaluations/arc_easy_kev_ft
```

The evaluator reads the checkpoint identity from `.env`, rejects the released base model, sends HTTP decisions to the local server, and writes the reports. Without the server, requests fail with connection errors.

### Test Case 3: Kev on ARC-Challenge — Smoke Test

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_challenge_test_full/ARC_challenge_test.csv \
  --dataset-config challenge --split test \
  --decision-model kev \
  --output-dir outputs/cot_evaluations/arc_challenge_kev_ft \
  --limit 10
```

### Test Case 4: Laya on ARC-Challenge

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_challenge_test_full/ARC_challenge_test.csv \
  --dataset-config challenge --split test \
  --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_arc_easy/checkpoint \
  --output-dir outputs/cot_evaluations/arc_challenge_laya_ft
```

Using an ARC-Easy checkpoint on ARC-Challenge is a transfer evaluation; keep a separate output directory.

### Test Case 5: JEV — Smoke Test

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv \
  --dataset-config easy --split test \
  --decision-model jev \
  --limit 2 \
  --output-dir outputs/cot_evaluations/arc_easy_jev_smoke
```

Verify authentication, valid predictions, and the actual returned model version before the full run. Two complete rows require up to eight initial API calls.

### Test Case 6: JEV — Full Sequential Evaluation

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv \
  --dataset-config easy --split test \
  --decision-model jev \
  --output-dir outputs/cot_evaluations/arc_easy_jev
```

### Test Case 7: Laya on LogiQA (Auto-Detected)

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/LogiQA_default_test.csv \
  --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_arc_easy/checkpoint \
  --output-dir outputs/cot_evaluations/logiqa_laya_ft
```

The dataset (`logiqa`), config (`default`), and split (`test`) are auto-detected from the CSV name. Context is embedded into each test condition's state.

### Test Case 8: Kev or Clef on Typed Decisions

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
  --decision-model kev \
  --kev-finetuned-checkpoint outputs/finetunes/kev_arc_easy/checkpoint \
  --output-dir outputs/cot_evaluations/typed_decisions_kev_ft
```

Environment state is preserved across question-only and all three CoT reasoning variants.

### Test Case 9: JEV on LogiQA or Typed Decisions with Concurrency

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/LogiQA_default_test.csv \
  --decision-model jev \
  --output-dir outputs/cot_evaluations/logiqa_jev \
  --concurrency 10
```

Runs hosted JEV with 10 concurrent HTTP worker threads against the TypeSafe API.

### Test Case 10: Fine-Tuned Model on ProofWriter

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/ProofWriter_default_test.csv \
  --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_arc_easy/checkpoint \
  --output-dir outputs/cot_evaluations/proofwriter_laya_ft
```

Theory statements are embedded in the prompt state, evaluating choices (`True`, `False`, `Unknown`).

## Installing and Using Concurrency

Download `enable_cot_concurrency.py` into the repository root. In that installer, set the target to the documented package location:

`TARGET = Path('src/arc_decision/cot_evaluation.py')`

If your checkout differs, locate the actual file first:

`find . -type f -name cot_evaluation.py`

Set `TARGET` to that result, then run from the repository root:

```bash
python enable_cot_concurrency.py
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py --help
```

Confirm that help lists `--concurrency`. The installer checks patch anchors, preserves existing JEV code, and saves a `.py.before_concurrency` backup. A failed validation leaves the evaluator unchanged. Successful installation and authenticated concurrent execution still need to be verified in your environment.

The patch uses bounded HTTP workers, keeps the original four-condition order within each row, and sends ledger writes to the main thread. It retains cache/retry logic and adds randomized retry jitter. It does not implement API-specific `Retry-After` handling or guarantee compliance with TypeSafe's rate limits.

Concurrency is **not multi-state API batching**: 50 workers means up to 50 separate calls in flight. Start with 10, then increase to 20 or 50 if the API permits. Values 60, 70, and 80 are configurable, but may cause rate-limit errors. Laya requires concurrency 1.

### Test Case 7: Resume JEV with Concurrency

Stop the sequential run with Ctrl+C before starting another process against the same output directory. After the patch succeeds:

```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv \
  --dataset-config easy --split test \
  --decision-model jev \
  --output-dir outputs/cot_evaluations/arc_easy_jev \
  --concurrency 10 \
  --resume
```

Change 10 to 50/60/70/80 as appropriate. Omit `--resume` for a new empty output directory. Do not change the CSV, model identity, or other cache-defining inputs when intending to reuse prior results. A switch from `jev-latest` to a pinned ID changes the identity; use a separate output directory for the pinned experiment.

The progress bar counts CSV rows, not HTTP calls. With 2,095 complete rows, there are up to 8,380 initial requests plus retries. Missing rationale conditions are skipped. During interruption, in-flight calls may need to finish before the patched process exits and flushes its ledger; avoid force-killing when possible.

## Execution Flow

1. Load `.env`, configure Python, and parse arguments.
2. Resolve model identity: local Laya checkpoint, Kev/Clef configured fine-tuned identity, or hosted JEV ID.
3. Load CSV records and rationale variants.
4. Create the experiment configuration, run store, and model client.
5. Evaluate each available condition, sequentially by default or with bounded HTTP concurrency after patch installation.
6. Record attempts and preserve resume information.
7. Write analysis CSVs/JSON and print the result summary.

## Output Artifacts

| File | Description |
|---|---|
| `decision_predictions.csv` | Example-condition predictions, correctness, confidence, probabilities, latency, tokens, and errors. |
| `summary.csv` | Condition accuracy, coverage, shared-completion accuracy, bootstrap difference against question-only with 95% CI, and input-token statistics. |
| `token_latency_summary.csv` | Condition latency and token-use mean, median, and p95. |
| `per_question_comparison.csv` | Four conditions side by side, including reasoning, prediction, and correctness. |
| `errors.csv` | Non-success attempts with error type/message. |
| `analysis/paired_comparisons.json` | Pairwise bootstrap comparisons: 2,000 samples, seed 7. |
| `attempts/decision_attempts.jsonl` | Append-only attempt ledger for resumability. |
| `manifest.json` | Run configuration and CSV metadata. |
| `dataset_records.jsonl` | Normalized ARC records. |
| `dataset_manifest.json` | Dataset fingerprint/version information. |

Each model writes its own reports. These commands do not generate a combined cross-model comparison report. Keep the same dataset and reasoning inputs across models and record JEV's actual version alongside the local checkpoint identities.