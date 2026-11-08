# Esha Ishaq · Digital Clone (Streamlit)

> **This is an AI clone, not the real person.** It answers only from notes Esha Ishaq wrote for this assignment.

Live app for Assignment 2 (Hybrid RAG). The full submission, including the report, the Gradio version and the evaluation scripts, is on Hugging Face:
https://huggingface.co/spaces/bhonduu/esha-ishaq-digital-clone

Pipeline: `data/*.md` → ~250-word chunks with 40-word overlap → BM25 (`rank_bm25`) + dense (`all-MiniLM-L6-v2`, precomputed `embeddings.npy`) → Reciprocal Rank Fusion → Gemini answers in first person from the retrieved chunks only. The Gemini key is stored as the `GEMINI_API_KEY` Streamlit secret.
