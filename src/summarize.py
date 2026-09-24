"""
Summarise Inspect logs from ccts_eval.py into one table with bootstrap CIs.

    python src/summarize.py logs/            # every .eval log in the folder
    python src/summarize.py logs/x.eval      # one log

Scores: C = 1, P = 0.5, I = 0. Also reports, for the false-premise task,
the rate at which the model *accepted* the premise (grade I). That is the
number to watch.
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

VAL = {"C": 1.0, "P": 0.5, "I": 0.0}


def boot(xs, n=2000, seed=0):
    if not xs:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n)]


def rows_from(path):
    log = read_eval_log(str(path))
    model = log.eval.model
    task = log.eval.task.split("/")[-1]
    for s in log.samples or []:
        score = next(iter(s.scores.values())) if s.scores else None
        grade = score.value if score else None
        if isinstance(grade, (int, float)):
            grade = {1.0: "C", 0.5: "P", 0.0: "I"}.get(float(grade))
        yield {"model": model, "task": task, "id": s.id,
               "topic": s.metadata.get("topic"), "grade": grade}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    p = Path(args.path)
    files = [p] if p.is_file() else [Path(l.name.replace("file://", "")) for l in list_eval_logs(str(p))]
    rows = [r for f in files for r in rows_from(f) if r["grade"] in VAL]
    if not rows:
        sys.exit("No scored samples found.")

    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["task"])].append(VAL[r["grade"]])
    print(f"\n{'model':<34}{'task':<22}{'n':>4}{'score':>8}   95% CI")
    for (m, t), xs in sorted(by.items()):
        mean, lo, hi = boot(xs)
        print(f"{m:<34}{t:<22}{len(xs):>4}{mean:>8.2f}   [{lo:.2f}, {hi:.2f}]")

    fp = [r for r in rows if r["task"] == "ccts_false_premise"]
    if fp:
        acc = [1.0 if r["grade"] == "I" else 0.0 for r in fp]
        mean, lo, hi = boot(acc)
        print(f"\nPremise acceptance rate (answered inside a false premise): "
              f"{mean:.0%}  [{lo:.0%}, {hi:.0%}]  n={len(acc)}")

    topics = defaultdict(list)
    for r in rows:
        topics[r["topic"]].append(VAL[r["grade"]])
    print("\nBy topic:")
    for k, xs in sorted(topics.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {k:<16}{sum(xs)/len(xs):.2f}  (n={len(xs)})")


if __name__ == "__main__":
    main()
