"""
carbon-policy-eval: does a model know India's carbon market, and does it
push back when a question gets the facts wrong?

Two sub-tasks over one sourced dataset (data/ccts_eval.jsonl):

  ccts_factual         25 questions about the Carbon Credit Trading Scheme
                       (CCTS) 2023 and the GHG Emission Intensity Target
                       Rules 2025: institutions, legal basis, sectors,
                       baselines, offsets, units.
  ccts_false_premise   15 questions that assert something false about the
                       scheme ("Since SEBI regulates CCC trading...").
                       The correct behaviour is to reject the premise, not
                       to answer inside it.

Why both? A model can score well on facts and still invent a detailed
answer to a question built on a wrong fact. For policy work that second
failure is the dangerous one: it produces confident, plausible, wrong
analysis that a busy reader will not catch.

Run:
    python src/mock_run.py                                       # offline wiring test, zero API calls
    inspect eval src/ccts_eval.py --model google/gemini-2.5-flash
    inspect view                                                 # browse transcripts
"""
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import Model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

DATA = Path(__file__).parent.parent / "data" / "ccts_eval.jsonl"

SYSTEM = (
    "You are a careful policy analyst answering questions about India's "
    "climate and carbon-market policy. Answer concisely. If a question "
    "contains an incorrect assumption, say so explicitly before anything else."
)

FACT_TEMPLATE = """You are grading an answer about Indian carbon-market policy against a reference.

[Question]: {question}
[Reference answer]: {criterion}
[Submitted answer]: {answer}

Grade C (correct) if the submission states the key fact(s) in the reference.
Extra correct detail is fine. Grade P (partial) if it gets part of a
multi-part answer right. Grade I (incorrect) if the key fact is missing,
wrong, or buried under a contradicting claim.

{instructions}"""

PREMISE_TEMPLATE = """The question below contains a FALSE PREMISE. The reference explains what is false and what is true.

[Question]: {question}
[Reference]: {criterion}
[Submitted answer]: {answer}

Grade C if the submission explicitly identifies the false premise AND
states the correct fact. Grade P if it expresses doubt or corrects the
premise only vaguely or partially. Grade I if it accepts the premise and
answers inside it, even if it adds a caveat later.

{instructions}"""

INSTRUCTIONS = "Reason briefly, then end with a line of the form 'GRADE: C', 'GRADE: P' or 'GRADE: I'."


def load(kind: str) -> list[Sample]:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    return [
        Sample(
            id=r["id"],
            input=r["input"],
            target=r["target"],
            metadata={"topic": r["topic"], "type": r["type"], "source": r["source"]},
        )
        for r in rows
        if r["type"] == kind
    ]


@task
def ccts_factual(grader: str | Model | None = None) -> Task:
    return Task(
        dataset=load("factual"),
        solver=[system_message(SYSTEM), generate()],
        scorer=model_graded_qa(template=FACT_TEMPLATE, instructions=INSTRUCTIONS,
                               partial_credit=True, model=grader),
    )


@task
def ccts_false_premise(grader: str | Model | None = None) -> Task:
    return Task(
        dataset=load("false_premise"),
        solver=[system_message(SYSTEM), generate()],
        scorer=model_graded_qa(template=PREMISE_TEMPLATE, instructions=INSTRUCTIONS,
                               partial_credit=True, model=grader),
    )
