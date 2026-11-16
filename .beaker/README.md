# Beaker: BizAgentBench single-turn

Optimizes only `payment/data/knowledge` (the documents the support agent reads).
The simulated bank, task contracts and verifier are fixed.

## How a case runs

The original benchmark simulates the customer with an LLM over many turns. Here
each case is one scripted customer message: the contract's opening plus its
case-specific facts (`build_dataset.py` drops simulator boilerplate by frequency).
The agent works with its tools until it replies in text. A rule-based customer then
approves every pending request, presses the portal's wait button if the agent asks,
or ends the conversation. File uploads are not scripted.

## Score

Expected results come from running the verifier on each untouched world. A case scores
the share of its initially failing checks that pass afterwards, minus checks that
passed initially and fail afterwards. Fee limits are scored by the verifier's own checks.
Approving every request means an over-limit proposal is executed and fails its check.

## Agent model

With a selected model Beaker's gateway is used. Otherwise set `AGENT_MODEL_ID`
(required) and optionally `AGENT_BASE_URL` (default OpenAI) and `AGENT_API_KEY`.

## Dataset

`uv run python .beaker/upload_dataset.py` uploads 38 train and 12 test cases.
