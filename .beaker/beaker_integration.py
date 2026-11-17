"""Optimize the BizAgentBench payment-support agent on the benchmark's own episodes.

Each case is a full customer conversation, run with the benchmark's LLM customer.
The score is the share of the verifier checks that fail on the untouched world and
pass after the agent works, minus any check that passed at the start and fails afterwards.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
)
from pydantic import BaseModel

# At 50 at once 37 of 38 baseline workers were SIGKILLed (out of memory); 25 is untested.
CONCURRENT_CASE_LIMIT = 25
# The benchmark's 5400s episode limit plus time to start and read the episode back.
CASE_TIMEOUT_SECONDS = 5700
MAX_STDERR_ARTIFACT_CHARS = 20_000
DEFAULT_AGENT_BASE_URL = "https://api.openai.com/v1"
# Only the documents the agent reads; the simulated bank, contracts and verifier stay fixed.
EDITABLE_SOURCE_PATHS = ("payment/data/knowledge",)

logger = logging.getLogger(__name__)
concurrent_case_limit = asyncio.Semaphore(CONCURRENT_CASE_LIMIT)


class Row(BaseModel):
    id: str
    input: dict
    expected: dict


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        yield Case(id=row.id, input=row.input, expected=row.expected)


def agent_model_environment(runtime: RolloutRuntime) -> dict[str, str]:
    """Route the agent through Beaker's gateway for a selected model, else the configured endpoint."""
    if runtime.model:
        target = inference_target(runtime)
        return {
            "AGENT_BASE_URL": target.base_url,
            "AGENT_API_KEY": target.api_key,
            "AGENT_MODEL_ID": target.model,
        }
    return {
        "AGENT_BASE_URL": os.environ.get("AGENT_BASE_URL", DEFAULT_AGENT_BASE_URL),
        "AGENT_MODEL_ID": os.environ["AGENT_MODEL_ID"],
        **({"AGENT_API_KEY": os.environ["AGENT_API_KEY"]} if os.environ.get("AGENT_API_KEY") else {}),
    }


def log_case_failure(case_input: dict, message: str) -> RuntimeError:
    # Beaker only keeps case errors once a candidate exists; the log shows them for any run.
    logger.error("case %s: %s", case_input.get("task_id"), message[:2000])
    return RuntimeError(message)


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    if not isinstance(case_input, dict):
        raise ValueError("Expected a BizAgentBench case input object")
    worker_path = Path(__file__).with_name("run_case_worker.py")
    queued_at = time.monotonic()
    async with concurrent_case_limit:
        started_at = time.monotonic()
        # A fresh interpreter keeps every case's simulated world separate.
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(worker_path),
            json.dumps(case_input),
            env={**os.environ, **agent_model_environment(runtime)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=CASE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()
        except BaseException:
            process.kill()
            await process.wait()
            raise
    timings = {
        "queued_seconds": round(started_at - queued_at, 1),
        "run_seconds": round(time.monotonic() - started_at, 1),
    }
    error_text = stderr.decode(errors="replace").strip()
    runtime.trace.artifact(
        "worker_stderr",
        error_text[-MAX_STDERR_ARTIFACT_CHARS:],
        media_type="text/plain",
        filename="worker_stderr.txt",
    )
    if timed_out:
        raise log_case_failure(case_input, f"Case timed out after {CASE_TIMEOUT_SECONDS}s {timings}")
    if process.returncode:
        hint = " (SIGKILL: likely out of memory)" if process.returncode == -9 else ""
        raise log_case_failure(
            case_input, f"Case worker failed (returncode={process.returncode}{hint}): {error_text[-4000:]}\n{timings}"
        )
    outcome = json.loads(stdout.decode().strip().splitlines()[-1])
    trace_outcome(runtime, outcome, timings)
    if "error" in outcome:
        raise log_case_failure(case_input, f"Case failed: {outcome['error']}\n{timings}")
    return CaseResult(
        output={
            "timings": {**timings, **model_call_statistics(outcome["model_calls"])},
            "episode_stats": outcome["episode_stats"],
            "checks": outcome["checks"],
            "goal_success": outcome["goal_success"],
            "process_success": outcome["process_success"],
            "protection_success": outcome["protection_success"],
            "strict_success": outcome["strict_success"],
            "agent_tool_calls": [
                call["name"]
                for message in outcome["conversation"]
                for call in message.get("tool_calls") or []
                if call["requestor"] == "assistant"
            ],
        },
        output_kind="record",
    )


def model_call_statistics(model_calls: list[dict]) -> dict:
    """Where a case's time went, kept in the case record so slow cases are visible without traces."""
    seconds = [call["wall_seconds"] for call in model_calls]
    return {
        "model_calls": len(model_calls),
        "model_seconds": round(sum(seconds), 1),
        "slowest_model_call_seconds": round(max(seconds, default=0), 1),
        "input_tokens": sum(call["usage"].get("prompt_tokens") or 0 for call in model_calls),
    }


def trace_outcome(runtime: RolloutRuntime, outcome: dict, timings: dict) -> None:
    """Replay the worker's model and tool calls into the Beaker trace."""
    trace = runtime.trace
    conversation = outcome.get("conversation", [])
    trace.artifact(
        "conversation", json.dumps(conversation), media_type="application/json", filename="conversation.json"
    )
    with trace.stage("simulation", attributes=timings):
        for call in outcome.get("model_calls", []):
            with trace.model_call(
                operation=call["call_name"],
                provider=call["model"].partition(":")[0] if ":" in call["model"] else "openai",
                model=call["model"],
                input_messages=call["input_messages"],
            ) as span:
                span.attribute("wall_seconds", call["wall_seconds"])
                span.output(call["output"])
                usage = call["usage"]
                # Without cached tokens Beaker prices every prompt token at the full input rate.
                span.usage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                )
        tool_results = {
            message["id"]: message["content"] for message in conversation if message["role"] == "tool"
        }
        for message in conversation:
            for call in message.get("tool_calls") or []:
                with trace.tool_call(call["name"], arguments=call["arguments"], call_id=call["id"]) as span:
                    span.output(tool_results.get(call["id"]))


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    outcome = result.output
    if not isinstance(outcome, dict) or "checks" not in outcome:
        raise ValueError("Case did not return verifier checks")
    final_passed = {check["id"]: check["passed"] for check in outcome["checks"]}
    failing_at_start = [check["id"] for check in case.expected["failing_checks"]]
    passing_at_start = [check["id"] for check in case.expected["passing_checks"]]
    regressed = [check_id for check_id in passing_at_start if final_passed.get(check_id) is False]
    fixed = [check_id for check_id in failing_at_start if final_passed.get(check_id) is True]
    checks = [
        Check(
            name=f"Fixes: {check_id}",
            verdict="pass" if check_id in fixed else "fail",
            message="Failed on the untouched world; must pass after the agent works",
        )
        for check_id in failing_at_start
    ] + [
        Check(
            name=f"Regressed: {check_id}",
            verdict="fail",
            message="Passed on the untouched world; the agent's actions broke it",
        )
        for check_id in regressed
    ]
    goal_progress = len(fixed) / len(failing_at_start)
    return CaseScore(
        objective=max(0.0, (len(fixed) - len(regressed)) / len(failing_at_start)),
        field_scores={
            "goal_progress": goal_progress,
            "kept_passing": 1 - len(regressed) / len(passing_at_start) if passing_at_start else 1.0,
            "process_success": float(outcome["process_success"]),
            "protection_success": float(outcome["protection_success"]),
            "strict_success": float(outcome["strict_success"]),
        },
        checks=tuple(checks),
    )


integration = Integration(
    targets=repository(EDITABLE_SOURCE_PATHS),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
