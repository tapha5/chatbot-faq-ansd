"""
sample_facts.py
----------------
Affiche quelques chunks par chapitre pour t'aider à rédiger
les 20 questions-réponses d'évaluation à partir de faits réels du rapport.

Usage:
    python src/sample_facts.py
"""

import json
import random
from collections import defaultdict

with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

by_file = defaultdict(list)
for c in chunks:
    by_file[c["source_file"]].append(c)

random.seed(42)  # reproductible

for source, file_chunks in by_file.items():
    print(f"\n{'='*80}\n{source} ({len(file_chunks)} chunks)\n{'='*80}")
    sample = random.sample(file_chunks, min(5, len(file_chunks)))
    for c in sample:
        print(f"\n--- page {c['page']} ---")
        print(c["text"][:400])