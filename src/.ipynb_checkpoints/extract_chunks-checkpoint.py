"""
extract_chunks.py
------------------
Etape 1 du pipeline RAG : extraction du texte depuis le(s) PDF(s)
et découpage en chunks (paragraphes) pour l'indexation.

Usage:
    python src/extract_chunks.py --input data/rapport_ansd.pdf --output data/chunks.json
"""

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extrait le texte page par page. Retourne une liste de dicts {page, text}."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def clean_text(text: str) -> str:
    """Nettoie le texte extrait (espaces multiples, retours à la ligne cassés)."""
    text = re.sub(r"\s+", " ", text)          # normalise les espaces/retours ligne
    text = re.sub(r"\.{3,}", ".", text)         # supprime les longues séries de points (sommaires)
    return text.strip()


def chunk_text(pages: list[dict], chunk_size: int = 400, overlap: int = 60) -> list[dict]:
    """
    Découpe le texte en chunks de ~chunk_size mots avec overlap.
    Chaque chunk garde une trace de la/les page(s) source pour la citation.
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = clean_text(page_data["text"])
        if len(text) < 50:  # page quasi vide (souvent tableaux/graphiques)
            continue

        words = text.split(" ")
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)
            if len(chunk_str.strip()) > 30:
                chunks.append({
                    "id": chunk_id,
                    "page": page_num,
                    "text": chunk_str,
                    "n_words": len(chunk_words),
                })
                chunk_id += 1
            start += chunk_size - overlap  # overlap pour ne pas couper une idée

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin du PDF (ou dossier de PDFs)")
    parser.add_argument("--output", default="data/chunks.json")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=60)
    args = parser.parse_args()

    input_path = Path(args.input)
    pdf_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"Extraction de {pdf_file.name}...")
        pages = extract_text_from_pdf(str(pdf_file))
        chunks = chunk_text(pages, args.chunk_size, args.overlap)
        for c in chunks:
            c["source_file"] = pdf_file.name
        all_chunks.extend(chunks)

    # Réindexer les ids globalement
    for i, c in enumerate(all_chunks):
        c["id"] = i

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(all_chunks)} chunks écrits dans {args.output}")
    print(f"Taille moyenne : {sum(c['n_words'] for c in all_chunks) / len(all_chunks):.0f} mots")


if __name__ == "__main__":
    main()
