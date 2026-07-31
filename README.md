# Chatbot FAQ — Rapport ANSD (RGPH-5)

Projet final NLP — Sujet E : Chatbot FAQ sur un document.

Architecture RAG simple : retrieval (TF-IDF ou embeddings) + QA extractif (CamemBERT fine-tuné SQuAD-FR).

## 1. Récupérer le document (100+ pages)

Télécharger 3-4 chapitres du rapport RGPH-5 (ANSD) en PDF, par exemple :
- Chapitre 1 : État et structure de la population
- Chapitre 2 : Éducation
- Chapitre 3 : Économie
- Chapitre 9 : Ménages

Source : https://www.ansd.sn/rapports/rgph-5-2023

Placer les PDFs téléchargés dans `data/` (ex: `data/pdfs/chapitre1.pdf`, etc.)

Puis fusionner (optionnel, si tu veux un seul PDF pour la démo) ou traiter directement le dossier :

```bash
pip install pypdf
python -c "
from pypdf import PdfWriter, PdfReader
import glob
writer = PdfWriter()
for f in sorted(glob.glob('data/pdfs/*.pdf')):
    reader = PdfReader(f)
    for page in reader.pages:
        writer.add_page(page)
writer.write('data/rapport_ansd_complet.pdf')
"
```

## 2. Installation

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## 3. Pipeline

```bash
# Etape 1 : extraction + chunking
python src/extract_chunks.py --input data/rapport_ansd_complet.pdf --output data/chunks.json

# Etape 2 : test rapide du retriever + construction des index
python src/retriever.py

# Etape 3 : test du pipeline QA complet
python src/qa_pipeline.py

# Etape 4 : évaluation EM/F1 (remplir data/eval_questions.json avec tes 20 Q/R d'abord)
python src/evaluate.py
```

## 4. Lancer l'application

```bash
streamlit run app.py
```

## 5. Déploiement

Pour Hugging Face Spaces :
1. Créer un Space (SDK: Streamlit)
2. Pousser `app.py`, `src/`, `requirements.txt`, et `data/chunks.json` (+ index pré-calculés si possible pour accélérer le démarrage)
3. Le fichier PDF complet n'a pas besoin d'être sur le Space — seul `chunks.json` est nécessaire à l'app

## Structure du projet

```
rag-faq-ansd/
├── data/
│   ├── pdfs/                  # PDFs bruts téléchargés (non versionnés si trop lourds)
│   ├── chunks.json            # sortie de l'extraction
│   ├── eval_questions.json    # 20 Q/R manuelles pour l'évaluation
│   └── eval_results.json      # résultats EM/F1
├── src/
│   ├── extract_chunks.py      # extraction PDF + chunking
│   ├── retriever.py           # TF-IDF (baseline) + embeddings (avancé)
│   ├── qa_pipeline.py         # QA extractif RAG
│   └── evaluate.py            # EM / F1
├── notebooks/
│   └── exploration.ipynb      # notebook livrable (exploration, baseline, modèle avancé, analyse)
├── app.py                     # interface Streamlit
├── requirements.txt
└── README.md
```

## Points d'attention pour le barème

- **Exploration (4 pts)** : dans le notebook, analyser nb pages, distribution longueur des chunks, vocabulaire, langue.
- **Baseline (3 pts)** : `TfidfRetriever` seul, sans QA extractif — montrer que ça retourne déjà des passages pertinents.
- **Modèle avancé (5 pts)** : `EmbeddingRetriever` + `RagQA`, comparer avec la baseline sur les mêmes questions.
- **Évaluation (4 pts)** : bien remplir les 20 Q/R manuelles dans `eval_questions.json` à partir du **contenu réel** de tes chapitres choisis (les valeurs actuelles sont des exemples à vérifier/remplacer).
- **Analyse des erreurs** : documenter les cas où le chatbot se trompe ou répond avec une faible confiance — c'est très valorisé par l'énoncé.
