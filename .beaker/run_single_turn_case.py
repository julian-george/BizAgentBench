"""Run one single-turn BizAgentBench case in an isolated process.

The agent gets one scripted customer message and works until it stops calling
tools. A rule-based customer then approves open requests, or presses the portal's
wait button when asked, and the agent continues. The verifier scores the final world.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "third_party/tau2-bench/src")]
DATA_DIRECTORY = REPOSITORY_ROOT / "payment/data"

MAX_AGENT_MODEL_CALLS = 150
MAX_CUSTOMER_TURNS = 20
MODEL_REQUEST_TIMEOUT_SECONDS = 600
MODEL_REQUEST_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503, 504)
AGENT_SYSTEM_PROMPT_INTRO = (
    "You are the customer-support agent. In each turn either speak naturally in English "
    "to the customer or call tools. Do not wrap customer-facing text in JSON.\n\n"
)
AGENT_ASKS_FOR_UPLOAD = re.compile(r"\bupload", re.IGNORECASE)
AGENT_ASKS_TO_WAIT = re.compile(r"\b(wait|advance|next (bank )?(event|update))\b", re.IGNORECASE)
CUSTOMER_APPROVAL_MESSAGE = "I have approved the pending request(s) in my portal. Please continue."
CUSTOMER_WAIT_MESSAGE = "I pressed the update button in my portal. Please continue."

model_calls: list[dict] = []
# Why the conversation ended and what the scripted customer did, for diagnosing low scores.
conversation_stats = {
    "end_reason": None,
    "customer_approvals": 0,
    "customer_waits": 0,
    "agent_upload_requests": 0,
    "tool_errors": 0,
}


def request_chat_completion(messages: list[dict], tool_schemas: list[dict]) -> dict:
    base_url = os.environ["AGENT_BASE_URL"].rstrip("/")
    model = os.environ["AGENT_MODEL_ID"]
    request_body = {"model": model, "messages": messages, "tools": tool_schemas}
    api_key = os.environ.get("AGENT_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started_at = time.monotonic()
    for attempt in range(1, MODEL_REQUEST_ATTEMPTS + 1):
        response = httpx.post(
            f"{base_url}/chat/completions",
            json=request_body,
            headers=headers,
            timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code not in RETRYABLE_STATUS_CODES:
            break
        time.sleep(min(30, 2**attempt))
    if response.status_code != 200:
        raise RuntimeError(f"Model request failed ({response.status_code}): {response.text[:500]}")
    body = response.json()
    message = body["choices"][0]["message"]
    model_calls.append(
        {
            "call_name": "agent_turn",
            "model": model,
            "input_messages": deepcopy(messages),
            "output": message,
            "usage": body.get("usage") or {},
            "wall_seconds": round(time.monotonic() - started_at, 2),
            "attempt": attempt,
        }
    )
    print(
        f"model_call {len(model_calls)}: {model_calls[-1]['wall_seconds']}s attempt={attempt} "
        f"tool_calls={len(message.get('tool_calls') or [])}",
        file=sys.stderr,
        flush=True,
    )
    return message


def call_user_tool(environment, name: str, **arguments):
    from tau2.data_model.message import ToolCall

    response = environment.get_response(
        ToolCall(id=f"customer-{name}", name=name, arguments=arguments, requestor="user")
    )
    return json.loads(response.content)


def customer_reply(environment, agent_text: str) -> str | None:
    """Rule-based customer: approve what is open, wait only when asked, else end."""
    open_requests = [
        request
        for request in call_user_tool(environment, "get_my_requests").get("data") or []
        if request["status"] == "PENDING"
    ]
    for request in open_requests:
        call_user_tool(environment, "decide_request", request_id=request["id"], decision="APPROVE")
    conversation_stats["customer_approvals"] += len(open_requests)
    # Uploads are not scripted, so every request for one goes unanswered.
    conversation_stats["agent_upload_requests"] += bool(AGENT_ASKS_FOR_UPLOAD.search(agent_text))
    if open_requests:
        return CUSTOMER_APPROVAL_MESSAGE
    if AGENT_ASKS_TO_WAIT.search(agent_text):
        call_user_tool(environment, "wait_for_update")
        conversation_stats["customer_waits"] += 1
        return CUSTOMER_WAIT_MESSAGE
    return None


def run_agent_conversation(environment, customer_message: str, tool_schemas: list[dict]) -> list[dict]:
    from tau2.data_model.message import ToolCall

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT_INTRO + environment.get_policy()},
        {"role": "user", "content": customer_message},
    ]
    customer_turns = 0
    while len([call for call in model_calls if call["call_name"] == "agent_turn"]) < MAX_AGENT_MODEL_CALLS:
        agent_message = request_chat_completion(messages, tool_schemas)
        messages.append({k: v for k, v in agent_message.items() if v is not None})
        tool_calls = agent_message.get("tool_calls") or []
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call["function"]["arguments"] or "{}")
                    result = environment.get_response(
                        ToolCall(
                            id=tool_call["id"],
                            name=tool_call["function"]["name"],
                            arguments=arguments,
                            requestor="assistant",
                        )
                    ).content
                except json.JSONDecodeError:
                    result = json.dumps({"ok": False, "error": "Tool arguments were not valid JSON"})
                tool_failed = json.loads(result).get("ok") is False if result.startswith("{") else False
                conversation_stats["tool_errors"] += tool_failed
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
            continue
        customer_turns += 1
        follow_up = customer_reply(environment, agent_message.get("content") or "")
        if follow_up is None:
            conversation_stats["end_reason"] = "customer_had_nothing_to_do"
            break
        if customer_turns >= MAX_CUSTOMER_TURNS:
            conversation_stats["end_reason"] = "customer_turn_cap"
            break
        messages.append({"role": "user", "content": follow_up})
    else:
        conversation_stats["end_reason"] = "model_call_cap"
    return messages


def run_case(case_input: dict) -> dict:
    from payment.runtime.evaluation.runner import verify
    from payment.runtime.interaction.domain import get_environment
    from payment.runtime.tasks.contracts import read_contract

    contract = read_contract(case_input["task_id"], DATA_DIRECTORY)
    environment = get_environment(contract_snapshot=contract, directory=DATA_DIRECTORY)
    tool_schemas = [tool.openai_schema for tool in environment.get_tools()]
    conversation = run_agent_conversation(environment, case_input["customer_message"], tool_schemas)
    verification = verify(environment.tools.db.world, contract)
    return {
        "checks": [
            {"id": check["id"], "passed": check["passed"], "categories": check["categories"]}
            for check in verification["checks"]
        ],
        "goal_success": verification["goal_success"],
        "process_success": verification["process_success"],
        "protection_success": verification["protection_success"],
        "strict_success": verification["strict_success"],
        "conversation_stats": conversation_stats,
        "conversation": conversation,
    }


def main() -> None:
    try:
        outcome = run_case(json.loads(sys.argv[1]))
    except Exception:
        outcome = {"error": traceback.format_exc()[-4000:]}
    outcome["model_calls"] = model_calls
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
