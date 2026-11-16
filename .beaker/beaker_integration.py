"""Optimize the BizAgentBench payment-support agent on single-turn cases.

Each case is one scripted customer message. The score is the share of the
verifier checks that fail on the untouched world and pass after the agent works,
minus any check that passed at the start and fails afterwards.
"""

from __future__ import annotations

import asyncio
import json
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

# Several simultaneous interpreters, each holding a full world, can exhaust sandbox memory.
CONCURRENT_CASE_LIMIT = 8
CASE_TIMEOUT_SECONDS = 1800
MAX_STDERR_ARTIFACT_CHARS = 20_000
DEFAULT_AGENT_BASE_URL = "https://api.openai.com/v1"
# Only the documents the agent reads; the simulated bank, contracts and verifier stay fixed.
EDITABLE_SOURCE_PATHS = ("payment/data/knowledge",)

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


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    if not isinstance(case_input, dict):
        raise ValueError("Expected a BizAgentBench case input object")
    worker_path = Path(__file__).with_name("run_single_turn_case.py")
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
        raise RuntimeError(f"Case timed out after {CASE_TIMEOUT_SECONDS}s {timings}")
    if process.returncode:
        raise RuntimeError(
            f"Case worker failed (returncode={process.returncode}) {timings}\n{error_text[-4000:]}"
        )
    outcome = json.loads(stdout)
    trace_outcome(runtime, outcome, timings)
    if "error" in outcome:
        raise RuntimeError(f"Case failed {timings}\n{outcome['error']}")
    return CaseResult(
        output={
            "checks": outcome["checks"],
            "strict_success": outcome["strict_success"],
            "agent_tool_calls": [
                call["function"]["name"]
                for message in outcome["conversation"]
                for call in message.get("tool_calls") or []
            ],
        },
        output_kind="record",
    )


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
                provider="openai",
                model=call["model"],
                input_messages=call["input_messages"],
            ) as span:
                span.attribute("wall_seconds", call["wall_seconds"])
                span.attribute("attempt", call["attempt"])
                span.output(call["output"])
                usage = call["usage"]
                span.usage(
                    input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")
                )
        tool_results = {
            message["tool_call_id"]: message["content"]
            for message in conversation
            if message["role"] == "tool"
        }
        for message in conversation:
            for call in message.get("tool_calls") or []:
                with trace.tool_call(
                    call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"] or "{}"),
                    call_id=call["id"],
                ) as span:
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
