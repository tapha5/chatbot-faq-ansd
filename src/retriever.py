"""
retriever.py
------------
Etape 2-3 du pipeline RAG : indexation des chunks et recherche (retrieval).

Deux approches, comme demandé dans le barème :
  - TF-IDF + similarité cosinus  -> baseline classique (3 pts du barème)
  - Embeddings (sentence-transformers) -> modèle avancé (5 pts du barème)
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Baseline : TF-IDF + cosinus. Rapide, pas de GPU nécessaire."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words=None,  # on garde les mots-outils français, utiles pour le sens
        )
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({**self.chunks[idx], "score": float(scores[idx])})
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.vectorizer = data["vectorizer"]
        obj.matrix = data["matrix"]
        obj.chunks = data["chunks"]
        return obj


class EmbeddingRetriever:
    """Modèle avancé : embeddings multilingues + similarité cosinus."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, chunks: list[dict]):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model = SentenceTransformer(self.MODEL_NAME)
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({**self.chunks[idx], "score": float(scores[idx])})
        return results

    def save(self, path: str):
        np.save(path + "_emb.npy", self.embeddings)
        with open(path + "_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        from sentence_transformers import SentenceTransformer
        obj = cls.__new__(cls)
        obj.model = SentenceTransformer(cls.MODEL_NAME)
        obj.embeddings = np.load(path + "_emb.npy")
        with open(path + "_chunks.json", encoding="utf-8") as f:
            obj.chunks = json.load(f)
        return obj


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Petit test manuel
    chunks = load_chunks("data/chunks.json")

    print("Construction de l'index TF-IDF...")
    tfidf = TfidfRetriever(chunks)
    tfidf.save("data/tfidf_index.pkl")

    print("Construction de l'index embeddings (peut prendre quelques minutes)...")
    emb = EmbeddingRetriever(chunks)
    emb.save("data/emb_index")

    query = "Quelle est la population du Sénégal en 2023 ?"
    print("\n--- TF-IDF ---")
    for r in tfidf.search(query, top_k=3):
        print(f"[p.{r['page']}, score={r['score']:.3f}] {r['text'][:150]}...")

    print("\n--- Embeddings ---")
    for r in emb.search(query, top_k=3):
        print(f"[p.{r['page']}, score={r['score']:.3f}] {r['text'][:150]}...")
