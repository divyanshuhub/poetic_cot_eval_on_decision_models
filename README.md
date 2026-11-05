# ARC Gemini Reasoning to Decision Models

This package evaluates whether Gemini-generated visible rationales help a
selected decision model answer `allenai/ai2_arc` multiple-choice questions.
Each example produces independent normal and poetic Gemini rationales, then the
selected model receives one of three inputs:

1. question and choices only;
2. question, choices, and normal rationale;
3. question, choices, and poetic rationale.

The experiment supports `LAYA`, `Clef`, and `Kev` as separate run labels:

```text
--decision-model {laya,clef,kev}
```

Do not pool their results. Run one immutable output directory per model.

## Install

```bash
cd arc_decision_reasoning
/home/gpuuser7/gpuuser7_a/Divyanshu/GSM8K/.venv/bin/python -m pip install -e .
cp .env.example .env
```

Set the environment variable named by `gemini_api_key_env` in
`configs/experiment.yaml` for Gemini generation; it defaults to
`GEMINI_API_KEY`. Put the key value in `.env` or the shell environment, never in
the YAML file. To switch keys, set the alternate variable (for example,
`GEMINI_API_KEY_2`) in the environment and change `gemini_api_key_env` to that
name. No hosted decision-model API key is needed: Laya, Kev, and Clef are
Apache-2.0 open-weight models that run locally. Their models are deliberately
kept separate because their local runtimes differ. The launch scripts source
`.env` when it exists.

### CUDA launchers

Use `run_cuda.sh` for Python commands in this project. It prefixes
`LD_LIBRARY_PATH` with the CUDA library locations before Python starts, selects
the first visible GPU by default, and sets `LAYA_DEVICE=cuda`. It accepts normal
Python arguments, so run the CLI with `bash ./run_cuda.sh -m arc_decision.cli ...`.
Set `CUDA_VISIBLE_DEVICES` in `.env` if the A100 is not GPU 0.

### Laya: direct local runtime

Laya uses the local `laya` package and downloads `convaiinnovations/laya` on first
use. It runs in-process; no server is required.

```bash
python -m pip install -e '.[laya]'
# Optional: use Laya's typed-decision checkpoint rather than automatic routing.
export LAYA_CHECKPOINT=typed-decisions
```

Without `--laya-finetuned-checkpoint`, evaluation uses the original Laya router
(or the built-in route selected by `LAYA_CHECKPOINT`). To evaluate a local
fine-tuned output from `laya-train --out`, pass its directory explicitly:

```bash
bash ./run_cuda.sh -m arc_decision.cli evaluate \
  --output-dir outputs/arc_easy_test \
  --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_arc_easy/checkpoint
```

Relative checkpoint paths resolve from `arc_decision_reasoning/`. Use a separate
output directory when comparing base and fine-tuned Laya results. The selected
checkpoint is recorded in the manifest and decision predictions, and changes
the resume cache key. Run the evaluator through `bash ./run_cuda.sh -m
arc_decision.cli ...` so the CUDA library path is set before Laya loads.

### Kev: local SystemOne server

Kev's released adapter is `jaredpalmer/kev-4b`. Start its local server once, then
the evaluator uses `http://127.0.0.1:8008/v1/systemone` by default. The server
launcher detects the checkout containing both `pyproject.toml` and
`kev/serve.py`, including the nested `kev/kev` layout:

```bash
git clone https://github.com/jaredpalmer/kev.git ./kev
bash ./run_kev_server.sh
```

The model card describes a CUDA path with about 16 GB free GPU memory, or an MLX
path on Apple Silicon with 32 GB unified memory. Set `KEV_API_URL` only when your
local server uses a different loopback address or port. The launcher sets
`CUDA_VISIBLE_DEVICES` and `KEV_DTYPE=bf16`, and applies the CUDA library path to
the Python server process. Set `KEV_REPO_DIR` in `.env` only if the checkout is
elsewhere.

To evaluate the ARC fine-tuned Kev run, point `KEV_MODEL_ID` in
`arc_decision_reasoning/.env` to its checkpoint directory, then start the server
and evaluator with that same `.env`:

```text
KEV_MODEL_ID=/absolute/path/to/arc_decision_reasoning/outputs/finetunes/kev_arc_easy/checkpoint
```

The `.env.example` base-model value must be replaced for fine-tuned evaluation;
the evaluator intentionally refuses to silently use the released base model.

### Clef: local SystemOne server

Clef defaults to `Cloudflare/clef-flash`, the 9B variant, which is the practical
choice for an A100 80 GB. It exposes the local `/v1/systemone` contract when
served through the documented SGLang image. To use the larger 27B model, set
`CLEF_MODEL_ID=Cloudflare/clef` before starting the evaluator and server.

```bash
bash ./run_clef_server.sh
```

The launcher starts the documented SGLang container with GPU passthrough and
uses the model selected by `CLEF_MODEL_ID` in `.env` (Clef-Flash by default).
The container supplies its own CUDA libraries, so the host `LD_LIBRARY_PATH`
is only set for Python processes and is not injected into Docker.

The evaluator never contacts a hosted decision service. `KEV_API_KEY` and
`CLEF_API_KEY` are optional only when you add authentication to your own local
servers.

## Fine-tuning

The `scripts/finetune_*.py` launchers prepare any of the benchmark datasets
(`arc`, `typed_decisions`, `proofwriter`, `logiqa`) in each decision model's
native System One training format, then delegate training to the model's own
tool. Fine-tuning occurs strictly on the `train` split against gold/correct
answers (`gold_answer_label`). The official `validation` split is used for
monitoring; if a dataset has no official validation split (such as
`typed_decisions`), a deterministic seeded 10% holdout from `train` is used
instead. The `test` split is never used for training or monitoring.

Selectable via CLI parameters:
- `--dataset {arc, typed_decisions, proofwriter, logiqa}`: Choose benchmark dataset (default: `arc`).
- `--difficulty {easy, challenge}`: For ARC, maps to `ARC-Easy` or `ARC-Challenge`.
- `--dataset-config CONFIG`: Explicit configuration identifier (e.g., `ARC-Easy`, `default`).
- `--replace` / `--overwrite`: Cleans and replaces the target output directory if it already exists.
- `--base-model MODEL_OR_CHECKPOINT`: Replaces or warm-starts from a specific base model or fine-tuned checkpoint.

The default run directories are distinct per model, dataset, and configuration:

```text
outputs/finetunes/laya_<dataset>_<config_slug>/
outputs/finetunes/kev_<dataset>_<config_slug>/
outputs/finetunes/clef_<dataset>_<config_slug>/
```

Each contains `data/train.jsonl`, `data/validation.jsonl`, a dataset manifest,
the recorded native commands, validation-monitoring artifacts, and a
`checkpoint/` directory. Use `--prepare-only` to inspect generated data without
requiring a native trainer; `--dry-run` additionally prints the native commands
without starting them.

```bash
# Laya on ARC: native laya-train with validation evaluation.
python -m pip install -e '.[laya]'
bash ./run_cuda.sh scripts/finetune_laya.py --dataset arc --difficulty easy --prepare-only
bash ./run_cuda.sh scripts/finetune_laya.py --dataset arc --difficulty easy

# Kev on LogiQA: native kev.train followed by kev.benchmark on LogiQA validation.
git clone https://github.com/jaredpalmer/kev.git ./kev
bash ./run_cuda.sh scripts/finetune_kev.py --dataset logiqa --replace

# Clef on ProofWriter: native clef-finetune training.
python -m pip install clef-finetune
bash ./run_cuda.sh scripts/finetune_clef.py --dataset proofwriter --replace

# Replace / warm-start from a previous checkpoint:
bash ./run_cuda.sh scripts/finetune_laya.py --dataset arc --difficulty challenge \
  --base-model outputs/finetunes/laya_arc_easy/checkpoint --replace

# One wrapper for all three native paths:
bash ./run_cuda.sh scripts/finetune_decision_model.py \
  --model clef --dataset typed_decisions --prepare-only
```

For Kev, `--base-model` selects the warm-start checkpoint passed to `--init_from`,
while `--kev-base-model` selects the underlying architecture passed to `--base`.
The default `jaredpalmer/kev-4b` checkpoint automatically uses
`Qwen/Qwen3.5-4B-Base`. When changing to a different Kev checkpoint, set
`--kev-base-model` to that checkpoint's matching backbone.

The generated row shapes deliberately differ because the native trainers do:

```json
// Laya: laya-train reads expected answers at the row level.
{"state":{"question":"..."},"questions":{"arc_answer":{"type":"choice","instructions":"Select the correct answer for this ARC science question.","criteria":{"A":"...","B":"..."}}},"expected":{"arc_answer":"B"}}

// Kev: the JSONL line itself is a labelled System One request.
{"state":{"question":"..."},"questions":{"arc_answer":{"type":"choice","instructions":"Select the correct answer for this ARC science question.","criteria":{"A":"...","B":"..."},"label":"B"}}}

// Clef: clef-finetune calls the expected-answer mapping labels.
{"id":"arc_arc_easy__train__000000","state":{"question":"..."},"questions":{"arc_answer":{"type":"choice","instructions":"Select the correct answer for this ARC science question.","criteria":{"A":"...","B":"..."}}},"labels":{"arc_answer":"B"}}
```

For `logiqa`, reading context is preserved under `state.context`. For `proofwriter`, theory statements are preserved under `state.theory`. For `typed_decisions`, structured scenario environment keys are preserved in `state`.

`--native-arg=VALUE` may be repeated to pass a native training flag to the
selected tool. For Clef's checkpoint schedule, use `--clef-max-steps N` so the
wrapper can locate the final checkpoint for monitoring. Kev runs from
`KEV_REPO_DIR` (or `arc_decision_reasoning/kev`) using `uv run python -m
kev.train`; use `--kev-repo-dir` or `--kev-python` to override that setup.

## Local vLLM CoT Dataset

`scripts/generate_arc_cot_vllm.py` is independent of the Gemini experiment
pipeline. It loads ARC-Easy or ARC-Challenge from Hugging Face, submits the
question and ordered choices to `google/gemma-3-27b-it` through vLLM, and writes
three rationale variants per row: factual `standard_cot`, deliberately
misleading `poisoned_cot`, and five-line `poetic_cot`. Gold answer keys are
stored in the CSV but are never included in a generation prompt. `standard_cot`
ends with `The final answer is ####<choice_label>`; only its explanation portion
is passed to the poisoned and poetic prompts. Each newly generated poisoned CoT
is checked by a separate local model completion and saved only when that judge
selects a non-gold ARC option.

Install a vLLM build compatible with the host, then run from this project:

```bash
python -m pip install -e '.[vllm]'
bash ./run_cuda.sh scripts/generate_arc_cot_vllm.py \
  --type easy --split test --limit 5
```

The default output is
`outputs/vllm_cot/ARC_easy_test.csv`; the filename always follows
`ARC_<type>_<split>.csv`. Its columns are exactly `question`, `choices`,
`answerKey`, `standard_cot`, `poisoned_cot`, and `poetic_cot`. An existing CSV
requires an explicit mode: use `--resume` to fill blank CoT cells or `--force`
to regenerate all three CoT columns. The generator checkpoints after every vLLM
batch and updates a sibling `ARC_<type>_<split>_report.json` with run status,
complete/partial/empty row counts, and filled/blank cell counts for each CoT
column. The report is refreshed on checkpoints and written with current counts
when model startup or generation fails. Use `--type challenge`,
`--batch-size`, `--max-tokens`, `--tensor-parallel-size`, and
`--gpu-memory-utilization` to adapt the run to the selected hardware.

## Evaluate Fine-Tuned Models on vLLM CoTs

Use `scripts/evaluate_cot_decision_model.py` to evaluate one local fine-tuned
decision model against a vLLM-generated CSV. It runs four independent
conditions: question/choices only, standard CoT, poisoned CoT, and poetic CoT.
For `standard_cot`, the `The final answer is ####...` line is removed before the
model sees the explanation. Blank rationale cells are recorded as skipped for
their condition; question-only evaluation still runs.

```bash
# Laya: pass the native laya-train output directory.
bash ./run_cuda.sh scripts/evaluate_cot_decision_model.py \
  --input-csv outputs/vllm_cot_easy_test_full/ARC_easy_test_complete_rows.csv \
  --dataset-config easy --split test --decision-model laya \
  --laya-finetuned-checkpoint outputs/finetunes/laya_arc_easy/checkpoint \
  --output-dir outputs/cot_evaluations/arc_easy_laya_ft
```

For Kev and Clef, you can specify fine-tuned checkpoints directly via CLI
(`--kev-finetuned-checkpoint PATH` or `--clef-finetuned-checkpoint PATH`), or start the
corresponding local server with its fine-tuned model selected through `KEV_MODEL_ID`
or `CLEF_MODEL_ID` in `.env`, then use `--decision-model kev` or `--decision-model clef`.
The evaluator refuses the default released base-model IDs for these modes. Clef LoRA
checkpoints must be merged into a server-loadable model first. Pass `--dataset {arc, typed_decisions, proofwriter, logiqa}`
to evaluate across different dataset formats.

Each output directory contains `decision_predictions.csv`, `summary.csv`,
`token_latency_summary.csv`, `per_question_comparison.csv`, `errors.csv`, and
the normal attempt ledgers and run manifest. The summary uses the same accuracy,
coverage, shared-completion accuracy, and paired-bootstrap difference metrics
as the main ARC CLI. It compares each rationale condition with question-only,
and records all pairwise comparisons in `analysis/paired_comparisons.json`.
`summary.csv` also reports `cot_tokens_total`, `cot_tokens_mean`, and population
`cot_tokens_std` from decision-reported input-token usage. These are computed on
the shared four-condition cohort with valid usage counts; the percentage field
`cot_tokens_delta_percent_vs_question_only` compares each condition's mean to
the question-only mean. Missing usage is excluded, not counted as zero.
`--resume` retries failed requests and skips successful matching conditions.
Use a separate output directory for each decision model or checkpoint.

## Run

All CLI output directories are anchored to `arc_decision_reasoning`, regardless
of the current working directory. Relative `--output-dir` values resolve under
the project folder; absolute paths outside it are rejected. If `--output-dir`
is omitted, the CLI uses `output_dir` from `configs/experiment.yaml` (currently
`outputs/arc_easy_test`). An explicit `--output-dir` overrides that YAML value.

Start with a low-cost smoke sample:

```bash
bash ./run_cuda.sh -m arc_decision.cli run \
  --dataset-config easy \
  --split test \
  --limit 5 \
  --gemini-model <GEMINI_MODEL_ID> \
  --decision-model laya \
  --poem-form tanka \
  --output-dir outputs/arc_easy_laya_smoke
```

Stages can run independently:

```bash
bash ./run_cuda.sh -m arc_decision.cli prepare \
  --dataset-config challenge --split test \
  --output-dir outputs/arc_challenge_clef

bash ./run_cuda.sh -m arc_decision.cli generate \
  --gemini-model <GEMINI_MODEL_ID> --poem-form tanka \
  --output-dir outputs/arc_challenge_clef --resume

bash ./run_cuda.sh -m arc_decision.cli evaluate \
  --decision-model clef \
  --output-dir outputs/arc_challenge_clef --resume

bash ./run_cuda.sh -m arc_decision.cli analyze \
  --output-dir outputs/arc_challenge_clef
```

`normal` is accepted as an alias for ARC-Challenge. Gemini is always called one
question at a time, and a hard answer-leakage flag excludes a rationale from the
augmented decision condition unless `--include-flagged-reasoning` is explicitly
set for a sensitivity analysis. The Gemini request pace comes from
`gemini_requests_per_minute` in the YAML config and can be overridden with
`--gemini-requests-per-minute`; pacing does not increase the provider quota.
An explicit 429 quota/limit-exhaustion response is checkpointed and stops the
entire generation stage (and any composed `run`) before another request is sent.
Use `--resume` after the provider quota resets or is increased. Transient 429s
without an explicit exhaustion message retain the bounded retry behavior.

## Artifacts

Each output directory retains append-only `attempts/gemini_attempts.jsonl` and
`attempts/decision_attempts.jsonl` ledgers. Required derived projections are:

- `front_cot.csv`, containing `cot_reasoning` and `cot_output_tokens`;
- `front_poetic.csv`, containing `poetic_cot_reasoning` and
  `poetic_cot_output_tokens`;
- `decision_predictions.csv`, one row per example and condition;
- `summary.csv`, with accuracy/coverage and paired normal-versus-poetic Gemini
  output-token totals, means, exact deltas, and percentage change;
- `token_latency_summary.csv` and `per_question_comparison.csv`;
- `errors.csv` and `analysis/paired_comparisons.json`.

Summary token metrics use only examples with API-reported output counts for both
Gemini generations. Counts cover the full structured response, and deltas are
poetic minus normal; they are not rationale-text-only token counts.

`analyze` rebuilds all derived analysis files without Gemini or decision-model
calls. Secrets are redacted before an attempt record is written.

## Test

The test suite uses no API keys, network calls, or ARC downloads:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
