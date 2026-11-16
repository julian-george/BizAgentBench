# Beaker: BizAgentBench

Optimizes only `payment/data/knowledge` (the documents the support agent reads).
The simulated bank, task contracts and verifier are fixed.

## How a case runs

`run_case_worker.py` runs each case through `tools/evaluate.py`'s own episode runner:
the benchmark's orchestrator, LLM customer simulator (prompt and token limits included)
and 5400s episode limit. Each case runs in a fresh process so its simulated world stays separate.

## Score

Expected results come from running the verifier on each untouched world. A case scores
the share of its initially failing checks that pass afterwards, minus checks that
passed initially and fail afterwards.

## Models

- Agent: set `AGENT_MODEL_ID` (required) and optionally `AGENT_BASE_URL` (default OpenAI) and
  `AGENT_API_KEY`. With a selected comparison model Beaker's gateway is used instead.
- Customer: `scorer_model` in `beaker.yaml` (DeepSeek V4.1 Flash, as in the benchmark's
  example config), called through Beaker's gateway. Locally set `USER_BASE_URL`,
  `USER_API_KEY` and `USER_MODEL_ID` yourself.

## Dataset

`uv run python .beaker/upload_dataset.py` uploads 38 train and 12 test cases. Only each
row's `task_id` and expected checks are used.
