"""Upload the single-turn BizAgentBench cases as a Beaker dataset."""

import subprocess
import tempfile
from pathlib import Path

from build_dataset import build_rows, write_splits

DATASET_NAME = "bizagentbench-single-turn-v2"
AGENT_KEY = "bizagentbench-payment-support"

rows = build_rows()
with tempfile.TemporaryDirectory(prefix="bizagentbench-single-turn-") as directory:
    split_counts = write_splits(rows, Path(directory))
    subprocess.run(
        [
            "beaker", "dataset", "upload", directory,
            "--name", DATASET_NAME,
            "--agent", AGENT_KEY,
            "--total-count", str(sum(split_counts.values())),
            *[argument for split, count in split_counts.items() for argument in ("--split", f"{split}={count}")],
            "--json",
        ],
        check=True,
    )
