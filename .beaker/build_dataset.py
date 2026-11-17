"""Build the single-turn BizAgentBench dataset from the frozen task contracts.

The customer message is the contract's opening plus its case-specific facts,
and expected results come from running the verifier on each untouched world.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "third_party/tau2-bench/src")]

from payment.runtime.evaluation.runner import verify  # noqa: E402
from payment.runtime.tasks.contracts import read_contract  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "payment/data"
# A sentence shared by this many cases is simulator boilerplate, not a customer fact.
MAX_CASES_SHARING_A_FACT = 2
SPLIT_SEED = 42
# A smaller set split evenly keeps each round short and the held-out score less noisy.
CASE_COUNT = 48
# Leading labels and directives addressed to the old LLM customer, not facts about the business.
SIMULATOR_LABEL = re.compile(
    r"^(Your own (business )?facts[^:]*:|Your private facts are:|"
    r"The following private dossier is true, but never recite it wholesale:)\s*"
)
SIMULATOR_DIRECTIVE = re.compile(
    r"^(Play the actual|Disclose |Correct misunderstandings|Do not invent|You do not know|"
    r"Never |Reveal |Treat |Answer only|Approve |List and upload|After |You may use)"
)


def split_into_sentences(instructions: str) -> list[str]:
    sentences = []
    for line in instructions.splitlines():
        line = line.strip()
        if not line or line.startswith(("Disclose when", "Opening:", "Begin with:")):
            continue
        sentences.extend(re.split(r"(?<=[.!?])\s+", line))
    cleaned = [SIMULATOR_LABEL.sub("", sentence.removeprefix("- ").strip()) for sentence in sentences]
    return [sentence for sentence in cleaned if sentence and not SIMULATOR_DIRECTIVE.match(sentence)]


def customer_facts_by_case(contracts: dict[str, dict]) -> dict[str, list[str]]:
    sentences_by_case = {
        case_id: split_into_sentences(contract["instructions"])
        for case_id, contract in contracts.items()
    }
    cases_sharing = Counter(
        sentence for sentences in sentences_by_case.values() for sentence in set(sentences)
    )
    return {
        case_id: [
            sentence for sentence in sentences if cases_sharing[sentence] <= MAX_CASES_SHARING_A_FACT
        ]
        for case_id, sentences in sentences_by_case.items()
    }


def verifier_baseline(contract: dict) -> dict:
    result = verify(deepcopy(contract["initial_world"]), contract)
    return {
        "failing_checks": [
            {"id": check["id"], "categories": check["categories"]}
            for check in result["checks"]
            if not check["passed"]
        ],
        "passing_checks": [
            {"id": check["id"], "categories": check["categories"]}
            for check in result["checks"]
            if check["passed"]
        ],
    }


def build_rows() -> list[dict]:
    manifest = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
    contracts = {task["id"]: read_contract(task["id"], DATA_DIRECTORY) for task in manifest["tasks"]}
    facts_by_case = customer_facts_by_case(contracts)
    rows = []
    for case_id, contract in contracts.items():
        customer_message = contract["opening"]
        if facts_by_case[case_id]:
            customer_message += (
                "\n\nTo save time, here is everything relevant about my situation:\n"
                + "\n".join(f"- {fact}" for fact in facts_by_case[case_id])
            )
        rows.append(
            {
                "id": case_id,
                "input": {"task_id": case_id, "customer_message": customer_message},
                "expected": verifier_baseline(contract),
            }
        )
    return rows


def write_splits(rows: list[dict], output_directory: Path) -> dict[str, int]:
    sampled = random.Random(SPLIT_SEED).sample(rows, CASE_COUNT)
    test_count = len(sampled) // 2
    splits = {"train": sampled[test_count:], "test": sampled[:test_count]}
    output_directory.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in splits.items():
        with (output_directory / f"{split_name}.jsonl").open("w") as split_file:
            for row in split_rows:
                split_file.write(json.dumps(row) + "\n")
    return {split_name: len(split_rows) for split_name, split_rows in splits.items()}


if __name__ == "__main__":
    output_directory = Path(sys.argv[1])
    print(json.dumps(write_splits(build_rows(), output_directory)))
