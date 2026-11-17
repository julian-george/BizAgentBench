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
- Customer: DeepSeek V4.1 Flash on OpenRouter, the benchmark's default simulator. In a hosted run
  Beaker's sandbox proxy sends the `openrouter.ai` calls to its gateway, as it does the agent's
  OpenAI calls. Locally set `OPENROUTER_API_KEY` as `USER_API_KEY`, or point `USER_BASE_URL` and
  `USER_MODEL_ID` elsewhere. A scorer-model credential is not available to rollouts, so it can't be used.

## Dataset

`uv run python .beaker/upload_dataset.py` uploads 48 of the 50 cases as
`bizagentbench-single-turn-v2`: 24 train and 24 test. Only each row's `task_id` and expected checks are used.
