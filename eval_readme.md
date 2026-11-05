# CoT Decision Model Evaluation Guide (`eval_readme.md`)

This guide explains how to run four-condition Chain-of-Thought (CoT) evaluations across all fine-tuned decision models (**Laya**, **Kev**, **Clef**) and the hosted **JEV** API using `scripts/evaluate_cot_decision_model.py`.

---

## 1. Overview of the 4 Evaluation Conditions

Every example in the input CoT CSV is evaluated under four distinct conditions:

| Condition | Input Provided to Model | Description |
| :--- | :--- | :--- |
| `question_only` | Question + Choices | Baseline choice task with no reasoning. |
| `standard_reasoning` | Question + Choices + Standard CoT | Visible standard CoT rationale. The final answer line (e.g. `#### <label>`) is automatically stripped before inference so the model must rely on the reasoning itself. |
| `poisoned_reasoning` | Question + Choices + Poisoned CoT | Adversarial reasoning designed to lead toward an incorrect distractor choice. |
| `poetic_reasoning` | Question + Choices + Poetic CoT | Metaphorical / poetic reasoning conveying the correct answer without verbatim leak. |

---

## 2. Supported Datasets & Input CSV Files

The runner supports multiple reasoning datasets and automatically detects dataset name and config subset from the input file path:

| Dataset | Subset Config | Default Input CSV Path |
| :--- | :--- | :--- |
| **Typed Decisions** | `all` | `outputs/vllm_cot/TypedDecisions_all_test.csv` |
| **ARC** | `easy` | `outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv` |
| **ARC** | `challenge` | `outputs/vllm_cot/ARC_challenge_test.csv` |
| **LogiQA** | `default` | `outputs/vllm_cot/LogiQA_default_test.csv` |
| **ProofWriter** | `default` | `outputs/vllm_cot/ProofWriter_default_test.csv` |

Each CSV contains: `question`, `choices` (JSON array), `answerKey`, `standard_cot`, `poisoned_cot`, and `poetic_cot`. Blank or missing rationales are recorded as skipped in that condition without failing the entire row.

---

## 3. Decision Model Architectures & How to Run

### Summary of Models

| Model | Architecture | Local Server Required? | Default Endpoint / Checkpoint |
| :--- | :--- | :--- | :--- |
| **Laya** | Direct local in-process | No | `outputs/finetunes/laya_<dataset>_<config>/checkpoint` |
| **Kev** | Local SystemOne HTTP | **Yes** (port `8008`) | `outputs/finetunes/kev_<dataset>_<config>/checkpoint` |
| **Clef** | Local SGLang Docker | **Yes** (port `30000`) | `outputs/finetunes/clef_<dataset>_<config>/checkpoint` |
| **JEV** | Hosted Cloud API | No | `https://api.typesafe.ai/v1/systemone` |

---

### A. Laya (Direct In-Process Model)

Laya runs locally in Python using your GPU. No background server process is needed.

#### Auto-discovery
If `--laya-finetuned-checkpoint` is omitted, the evaluator automatically detects the matching checkpoint in `outputs/finetunes/`:
```bash
cd ~/Divyanshu/GSM8K/arc_decision_reasoning

bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
  --decision-model laya \
  --output-dir outputs/cot_evaluations/typed_decisions_laya_ft
```

#### Explicit Checkpoint Path
```bash
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
  --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_typed_decisions_all/checkpoint \
  --output-dir outputs/cot_evaluations/typed_decisions_laya_ft
```

---

### B. Kev (Local SystemOne HTTP Server)

Kev is a client-server model that serves `/v1/systemone` on port `8008`.

#### Step 1: Start the Kev Server (Terminal 1)
Pass the checkpoint path to `run_kev_server.sh`:
```bash
cd ~/Divyanshu/GSM8K/arc_decision_reasoning
export PATH="$HOME/.local/bin:$PATH"

bash ./run_kev_server.sh outputs/finetunes/kev_typed_decisions_all/checkpoint
```
Wait until you see:
```text
serving /.../checkpoint (...) on cuda via torch (bfloat16) 127.0.0.1:8008
INFO:     Uvicorn running on http://127.0.0.1:8008 (Press CTRL+C to quit)
```

#### Step 2: Run the Evaluator (Terminal 2)
In another terminal while the server is running:
```bash
cd ~/Divyanshu/GSM8K/arc_decision_reasoning

bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
  --decision-model kev \
  --output-dir outputs/cot_evaluations/typed_decisions_kev_ft
```

---

### C. Clef (Local SGLang Server)

Clef runs inside a Docker SGLang container exposing port `30000`.

#### Step 1: Start the Clef Server (Terminal 1)
Set `CLEF_MODEL_ID` in `.env` to your checkpoint path, then run:
```bash
cd ~/Divyanshu/GSM8K/arc_decision_reasoning
bash ./run_clef_server.sh
```

#### Step 2: Run the Evaluator (Terminal 2)
```bash
cd ~/Divyanshu/GSM8K/arc_decision_reasoning

bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
  --decision-model clef \
  --output-dir outputs/cot_evaluations/typed_decisions_clef_ft
```

---

### D. JEV (Hosted TypeSafe Cloud API)

JEV is a cloud-hosted decision service. No local server, GPU setup, or checkpoint path is required.

Make sure `.env` contains:
```text
JEV_API_KEY=<your_api_key>
JEV_MODEL_ID=jev-latest
JEV_API_URL=https://api.typesafe.ai/v1/systemone
```

#### Run the Evaluator
```bash
cd ~/Divyanshu/GSM8K/arc_decision_reasoning

bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
  --decision-model jev \
  --output-dir outputs/cot_evaluations/typed_decisions_jev
```

> [!TIP]
> Use `--concurrency 4` or `--concurrency 8` with JEV to speed up evaluation over the network:
> ```bash
> bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
>   --input-csv outputs/vllm_cot/TypedDecisions_all_test.csv \
>   --decision-model jev \
>   --concurrency 4 \
>   --output-dir outputs/cot_evaluations/typed_decisions_jev
> ```

---

## 4. Fine-Tuned Checkpoint Naming Convention

Checkpoints generated during fine-tuning are stored under `outputs/finetunes/` with this standard structure:

```text
outputs/finetunes/{model}_{dataset}_{config_slug}/checkpoint
```

Examples:
- `outputs/finetunes/laya_typed_decisions_all/checkpoint`
- `outputs/finetunes/kev_typed_decisions_all/checkpoint`
- `outputs/finetunes/laya_arc_easy/checkpoint`
- `outputs/finetunes/kev_arc_easy/checkpoint`
- `outputs/finetunes/laya_logiqa_default/checkpoint`

---

## 5. Output Artifacts and Metrics

Every run generates the following artifacts in your `--output-dir`:

1. **`summary.csv`**:
   - `condition`: `question_only`, `standard_reasoning`, `poisoned_reasoning`, `poetic_reasoning`
   - `coverage`: Fraction of eligible examples successfully answered.
   - `accuracy`: Accuracy percentage under this condition.
   - `question_only_difference_pp`: Percentage point change compared to `question_only`.
   - `question_only_difference_ci_low_pp` / `ci_high_pp`: 95% bootstrap confidence interval.
   - `cot_tokens_mean`, `cot_tokens_total`, `cot_tokens_delta_percent_vs_question_only`: Token statistics.

2. **`decision_predictions.csv`**:
   - Per-example predictions, choice probabilities, confidence scores, latencies, and token counts.

3. **`summary.json`**:
   - Full machine-readable summary with all paired statistical comparisons:
     - `standard_reasoning_minus_question_only`
     - `poisoned_reasoning_minus_question_only`
     - `poetic_reasoning_minus_question_only`
     - `poisoned_reasoning_minus_standard_reasoning`
     - `poetic_reasoning_minus_poisoned_reasoning`

4. **`errors.jsonl`** & **`attempts.jsonl`**:
   - Ledger of any request failures or retries.

---

## 6. CLI Flags and Options Reference

| Option | Description |
| :--- | :--- |
| `--input-csv <path>` | **(Required)** Path to CoT CSV file. |
| `--decision-model <model>` | **(Required)** One of `laya`, `kev`, `clef`, `jev`. |
| `--output-dir <path>` | Destination directory for summary and prediction files. |
| `--limit <N>` | Evaluate only first `N` examples (ideal for smoke tests). |
| `--concurrency <N>` | Concurrent worker count (default: 1 for Laya, can be 4–8 for Jev/Kev). |
| `--resume` | Resume an interrupted run, skipping already completed rows. |
| `--force-regenerate` | Overwrite existing outputs in `--output-dir`. |
| `--max-retries <N>` | Maximum retry attempts for transient server errors (default: 5). |
| `--request-timeout <sec>` | Request timeout in seconds (default: 60.0). |
| `--laya-finetuned-checkpoint` | Explicit checkpoint path for Laya (auto-detected if omitted). |
| `--kev-finetuned-checkpoint` | Explicit checkpoint path for Kev (auto-detected if omitted). |
| `--clef-finetuned-checkpoint` | Explicit checkpoint path for Clef (auto-detected if omitted). |

