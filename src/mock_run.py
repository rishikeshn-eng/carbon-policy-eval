"""
Offline wiring test. Runs both tasks end to end with Inspect's mockllm
provider for the model *and* a scripted mock grader, so scoring, logging
and summarize.py can all be checked without an API key.

    python src/mock_run.py
    python src/summarize.py logs/

The grader cycles through C / P / I at fixed odds. That is a pipeline test,
not a result. Real numbers come from a real model.
"""
import random
import sys
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

sys.path.insert(0, str(Path(__file__).parent))
from ccts_eval import ccts_factual, ccts_false_premise  # noqa: E402


def scripted_grades(n=200, seed=7):
    rng = random.Random(seed)
    return [ModelOutput.from_content(model="mockllm/model",
                                     content=f"Mock reasoning.\nGRADE: {rng.choices('CPI', weights=[6, 2, 2])[0]}")
            for _ in range(n)]


if __name__ == "__main__":
    grader = get_model("mockllm/model", custom_outputs=scripted_grades(), memoize=False)
    eval([ccts_factual(grader=grader), ccts_false_premise(grader=grader)],
         model="mockllm/model", log_dir="logs", display="plain")
