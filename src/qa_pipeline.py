"""
qa_pipeline.py
--------------
Etape 4 du pipeline RAG : QA extractif.
Prend les chunks trouvés par le retriever et en extrait la réponse précise.

Modèle QA francophone : etalab-ia/camembert-base-squadFR-fquad-piaf
(fine-tuné sur FQuAD / SQuAD-FR, adapté au rapport ANSD en français)
"""

from transformers import pipeline


class RagQA:
    def __init__(self, retriever, qa_model_name: str = "etalab-ia/camembert-base-squadFR-fquad-piaf"):
        self.retriever = retriever
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

    def answer(self, question: str, top_k: int = 3, min_score: float = 0.05) -> dict:
        """
        1) Récupère les top_k chunks les plus pertinents (retrieval)
        2) Applique le modèle QA extractif sur chaque chunk
        3) Retourne la meilleure réponse avec sa source (page) et son score de confiance
        """
        candidates = self.retriever.search(question, top_k=top_k)

        best = None
        for chunk in candidates:
            if chunk["score"] < min_score:
                continue
            try:
                result = self.qa_pipeline(question=question, context=chunk["text"])
            except Exception:
                continue
            result["source_page"] = chunk["page"]
            result["source_text"] = chunk["text"]
            result["retrieval_score"] = chunk["score"]

            if best is None or result["score"] > best["score"]:
                best = result

        if best is None:
            return {
                "answer": "Je n'ai pas trouvé de réponse fiable dans le document.",
                "score": 0.0,
                "source_page": None,
                "source_text": None,
            }
        return best


if __name__ == "__main__":
    from retriever import TfidfRetriever, load_chunks

    chunks = load_chunks("data/chunks.json")
    retriever = TfidfRetriever(chunks)
    rag = RagQA(retriever)

    question = "Quelle est la population totale du Sénégal selon le RGPH-5 ?"
    result = rag.answer(question)
    print(f"Question : {question}")
    print(f"Réponse  : {result['answer']}")
    print(f"Confiance: {result['score']:.2f}")
    print(f"Source   : page {result['source_page']}")
