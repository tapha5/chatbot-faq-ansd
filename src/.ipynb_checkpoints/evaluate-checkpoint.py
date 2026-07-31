"""
evaluate.py
-----------
Etape 5 : évaluation rigoureuse avec Exact Match (EM) et F1,
selon la méthode standard SQuAD.

Prépare un fichier data/eval_questions.json avec 20 questions-réponses
manuelles, au format :
[
  {"question": "...", "answer": "..."},
  ...
]
"""

import json
import re
import string
from collections import Counter


def normalize_text(s: str) -> str:
    """Normalisation standard SQuAD : minuscules, ponctuation, articles, espaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(le|la|les|un|une|des|de|du|l)\b", " ", s)  # articles français
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(rag_qa, eval_path: str = "data/eval_questions.json") -> dict:
    with open(eval_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    em_scores, f1_scores = [], []
    details = []

    for item in eval_set:
        result = rag_qa.answer(item["question"])
        pred = result["answer"]
        em = exact_match(pred, item["answer"])
        f1 = f1_score(pred, item["answer"])
        em_scores.append(em)
        f1_scores.append(f1)
        details.append({
            "question": item["question"],
            "expected": item["answer"],
            "predicted": pred,
            "em": em,
            "f1": round(f1, 3),
            "confidence": round(result["score"], 3),
        })

    summary = {
        "n_questions": len(eval_set),
        "exact_match": round(100 * sum(em_scores) / len(em_scores), 1),
        "f1": round(100 * sum(f1_scores) / len(f1_scores), 1),
        "details": details,
    }
    return summary


if __name__ == "__main__":
    from retriever import TfidfRetriever, load_chunks
    from qa_pipeline import RagQA

    chunks = load_chunks("data/chunks.json")
    retriever = TfidfRetriever(chunks)
    rag = RagQA(retriever)

    results = evaluate(rag, "data/eval_questions.json")
    print(f"Exact Match : {results['exact_match']}%")
    print(f"F1          : {results['f1']}%")

    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
