import json

with open("data/eval_results.json", encoding="utf-8") as f:
    data = json.load(f)

for d in data["details"]:
    print("Q:", d["question"])
    print("  Attendu :", d["expected"])
    print("  Predit  :", d["predicted"], "| EM =", d["em"], "| F1 =", d["f1"], "| confiance =", d["confidence"])
    print()