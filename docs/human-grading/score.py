"""Rebuild the README's two hand-graded multi-turn rows from the blind grades: grades_max.json holds Max's verdict
for each shuffled reply number (1 correct, 0 not), and shuffle_key.json maps each number back to its model and
category. Run: python score.py"""
import json
from collections import defaultdict

key = {it["n"]: it for it in json.load(open("shuffle_key.json"))}
grades = {int(n): v for n, v in json.load(open("grades_max.json"))["grades"].items()}
assert sorted(grades) == sorted(key), "every reply graded exactly once"

tally = defaultdict(lambda: [0, 0])
for n, v in grades.items():
    it = key[n]
    for row in (it["category"], "overall"):
        tally[(it["model"], row)][0] += v
        tally[(it["model"], row)][1] += 1

models = sorted({it["model"] for it in key.values()})
for row in ("reference", "overall"):
    cells = []
    for m in models:
        k, n = tally[(m, row)]
        cells.append(f"{m} {round(100 * k / n)}% ({k}/{n})")
    print(f"{row}: " + ", ".join(cells))
