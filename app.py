"""
app.py
------
Interface Streamlit du chatbot FAQ sur le rapport ANSD (RGPH-5).

Lancement :
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from retriever import TfidfRetriever, EmbeddingRetriever, load_chunks
from qa_pipeline import RagQA

st.set_page_config(page_title="Chatbot FAQ - Rapport ANSD", page_icon="🇸🇳", layout="centered")

CHUNKS_PATH = "data/chunks.json"


@st.cache_resource(show_spinner="Chargement du modèle et de l'index...")
def load_rag(mode: str):
    chunks = load_chunks(CHUNKS_PATH)
    if mode == "TF-IDF (rapide)":
        retriever = TfidfRetriever(chunks)
    else:
        retriever = EmbeddingRetriever(chunks)
    return RagQA(retriever)


st.title("🇸🇳 Chatbot FAQ — Rapport ANSD (RGPH-5)")
st.caption("Posez une question sur le recensement général de la population du Sénégal (2023).")

with st.sidebar:
    st.header("Paramètres")
    mode = st.radio("Mode de recherche", ["TF-IDF (rapide)", "Embeddings (précis)"], index=0)
    top_k = st.slider("Nombre de passages consultés", 1, 10, 3)
    st.markdown("---")
    st.markdown(
        "**A propos**\n\n"
        "Ce chatbot utilise une architecture RAG (Retrieval-Augmented Generation) simple : "
        "recherche des passages pertinents dans le rapport, puis extraction de la réponse "
        "précise avec un modèle QA français."
    )

rag = load_rag(mode)

if "history" not in st.session_state:
    st.session_state.history = []

# Affichage de l'historique
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        conf = turn["confidence"]
        color = "green" if conf > 0.5 else "orange" if conf > 0.2 else "red"
        st.markdown(f":{color}[Confiance : {conf:.0%}]")
        if turn["source_page"]:
            with st.expander(f"📄 Source (page {turn['source_page']})"):
                st.write(turn["source_text"])

# Saisie utilisateur
question = st.chat_input("Votre question sur le rapport...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le document..."):
            result = rag.answer(question, top_k=top_k)

        st.write(result["answer"])
        conf = result["score"]
        color = "green" if conf > 0.5 else "orange" if conf > 0.2 else "red"
        st.markdown(f":{color}[Confiance : {conf:.0%}]")

        if result.get("source_page"):
            with st.expander(f"📄 Source (page {result['source_page']})"):
                st.write(result["source_text"])

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "confidence": result["score"],
        "source_page": result.get("source_page"),
        "source_text": result.get("source_text"),
    })
