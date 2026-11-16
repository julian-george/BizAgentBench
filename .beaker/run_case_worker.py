"""Run one BizAgentBench case in an isolated process, exactly as the benchmark runs it.

The agent and the LLM customer simulator are the two roles `tools/evaluate.py` runs,
with the benchmark's own orchestrator, customer prompt and token limits. This only
adds the endpoint configuration and a read-back of the finished episode.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "third_party/tau2-bench/src")]

# Limits and transport settings from configs/model.example.json.
EPISODE_TIMEOUT_SECONDS = 5400
MAX_TOOL_ERRORS = 16
AGENT_MAX_TOKENS = 262_144
CUSTOMER_MAX_TOKENS = 393_216
MODEL_REQUEST_TIMEOUT_SECONDS = 600
# The benchmark's default customer simulator. In a hosted run Beaker's sandbox proxy sends
# openrouter.ai calls to its gateway; OpenRouter may report the dated slug as the response model.
CUSTOMER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CUSTOMER_DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
CUSTOMER_RESPONSE_MODEL = "deepseek/deepseek-v4.1-flash-1017"


def role_config(
    role_prefix: str, max_tokens: int, *, default_base_url: str | None = None, default_model: str | None = None
) -> dict:
    return {
        "base_url": os.environ.get(f"{role_prefix}_BASE_URL", default_base_url),
        "api_key_env": f"{role_prefix}_API_KEY",
        "model": os.environ.get(f"{role_prefix}_MODEL_ID", default_model),
        "max_tokens": max_tokens,
        "timeout_seconds": MODEL_REQUEST_TIMEOUT_SECONDS,
        # The sandbox proxy's CA is only trusted through SSL_CERT_FILE, which httpx reads with trust_env.
        "trust_env": True,
        "extra_body": {},
        "send_seed": False,
        "temperature": None,
    }


def read_json(path: Path):
    return json.loads(path.read_text())


def read_episode(folder: Path) -> dict:
    """Model calls and conversation the benchmark saved for one finished episode."""
    ledger = [json.loads(line) for line in (folder / "api_calls.jsonl").read_text().splitlines()]
    model_calls = [
        {
            "call_name": "agent_turn" if row["role"] == "assistant" else "customer_turn",
            "model": row["model"],
            "input_messages": row["request"]["messages"],
            "output": row["response"]["choices"][0]["message"],
            "usage": row.get("usage") or {},
            "wall_seconds": round(row["duration_seconds"], 2),
        }
        for row in ledger
        if row.get("response")
    ]
    trajectory_path = next(
        path for path in (folder / "trajectory.json", folder / "trajectory_partial.json") if path.exists()
    )
    trajectory = read_json(trajectory_path)
    return {
        "model_calls": model_calls,
        "conversation": trajectory["messages"] if isinstance(trajectory, dict) else trajectory,
    }


def run_case(case_input: dict) -> dict:
    from loguru import logger
    from payment.runtime.interaction.domain import register
    from tools.evaluate import run_case as run_benchmark_case

    # tau2 logs every message at DEBUG; the parent buffers that stderr for all concurrent cases.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    register()
    config = {
        "seed": 0,
        "max_errors": MAX_TOOL_ERRORS,
        "episode_timeout_seconds": EPISODE_TIMEOUT_SECONDS,
        "agent": {**role_config("AGENT", AGENT_MAX_TOKENS), "compact_api_ledger": False},
        "user": {
            **role_config(
                "USER",
                CUSTOMER_MAX_TOKENS,
                default_base_url=CUSTOMER_DEFAULT_BASE_URL,
                default_model=CUSTOMER_DEFAULT_MODEL,
            ),
            "response_model": CUSTOMER_RESPONSE_MODEL,
            "compact_api_ledger": False,
        },
    }
    with tempfile.TemporaryDirectory() as output_directory:
        result = run_benchmark_case(case_input["task_id"], config, Path(output_directory))
        verification = result["verification"]
        if result["classification"] in ("infrastructure_invalid", "simulator_protocol_error"):
            raise RuntimeError(f"{result['classification']}: {result.get('error_class')}: {result.get('error')}")
        if "checks" not in verification:
            raise RuntimeError(f"Verifier failed: {result.get('classification')} {verification.get('error')}")
        episode = read_episode(Path(output_directory) / f"{case_input['task_id']}__trial0")
    return {
        "checks": [
            {"id": check["id"], "passed": check["passed"], "categories": check["categories"]}
            for check in verification["checks"]
        ],
        "goal_success": verification["goal_success"],
        "process_success": verification["process_success"],
        "protection_success": verification["protection_success"],
        "strict_success": verification["strict_success"],
        "episode_stats": {
            "termination": result["termination"],
            "classification": result["classification"],
            "tool_errors": result["tool_errors"],
            "tool_calls": result["actual_tool_calls"],
            "output_limit_roles": result["output_limit_roles"],
            "error": result.get("error"),
        },
        **episode,
    }


def main() -> None:
    try:
        outcome = run_case(json.loads(sys.argv[1]))
    except Exception:
        details = traceback.format_exc()
        # The last line names the exception; Beaker shortens long messages, so it goes first.
        outcome = {"error": f"{details.strip().splitlines()[-1]}\n{details[-4000:]}"}
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
