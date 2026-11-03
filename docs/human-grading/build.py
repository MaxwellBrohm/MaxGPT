"""Build a blind grading page from the ORIGINAL MaxGPT-3 multi-turn eval replies (multiturn_results_original.txt,
copied from the PC): all 96 replies, shuffled, model hidden. shuffle_key.json maps each reply to its model."""
import ast, json, random, re
from collections import Counter

txt = open("multiturn_results_original.txt", encoding="utf-8").read()
items = []
for b in txt.split("===MODEL=== ")[1:]:
    model = b.split("\n", 1)[0].strip()
    cid, cat = re.search(r"===CONV_ID=== (\S+) \((\w+)\)", b).groups()
    raw = re.search(r"===HISTORY=== (.*)\n", b).group(1).strip()
    try:
        hist = ast.literal_eval(raw)
    except Exception:
        hist = raw.strip("\"'").replace("\\n", "\n")
    check = re.search(r"===CONTEXT_CHECK=== (.*)\n", b).group(1)
    sample = re.search(r"===SAMPLE=== (\S+)", b).group(1)
    reply = b.split("===SAMPLE=== " + sample, 1)[1].split("===END===")[0].strip()
    items.append(dict(model=model, id=cid, category=cat, history=hist, check=check, sample=sample, reply=reply))
assert len(items) == 96, len(items)
random.Random(20261103).shuffle(items)
for i, it in enumerate(items):
    it["n"] = i + 1
json.dump(items, open("shuffle_key.json", "w"), indent=1)
blind = [dict(n=it["n"], history=it["history"], check=it["check"], reply=it["reply"]) for it in items]
page = open("template.html", encoding="utf-8").read().replace("__ITEMS__", json.dumps(blind))
open("grade.html", "w", encoding="utf-8").write(page)
print(Counter((it["model"], it["category"]) for it in items))
