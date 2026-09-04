# knowledge_base/embed_and_store.py
"""Embed PDFs and Open Food Facts data into ChromaDB.

This script performs the following steps:
1. Load all PDF files from `knowledge_base/raw/`.
2. Extract text from each PDF, apply the *section‑aware* chunking strategy
   defined in `knowledge_base/chunker.py`.
3. Load the filtered Open Food Facts CSV (`off_filtered.csv`) and turn each
   row into a single text chunk.
4. Generate embeddings for every chunk using the Sentence‑Transformers model
   `all‑MiniLM‑L6‑v2`.
5. Persist the embeddings in a local ChromaDB instance under two collections:
   - `regulations` – PDF chunks.
   - `products`    – OFF product chunks.
6. Print a short summary of the number of chunks embedded for each collection.

The script is intentionally idempotent – collections are created if they do not
exist, otherwise new data is added (duplicate IDs are avoided by using a
combination of source name and an incremental index).
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict

import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Import the section‑aware chunker from the same package
from .chunker import section_chunk

# Paths (relative to the repository root)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "knowledge_base" / "raw"
CHROMA_DB_DIR = REPO_ROOT / "knowledge_base" / "chroma_db"

# Ensure the persistence directory exists
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_pdfs(pdf_dir: Path) -> List[Dict]:
    """Load PDF files and return a list of chunk dictionaries.

    Each dictionary contains:
        - ``text``       : chunk text
        - ``metadata``   : dict with ``source`` (filename), ``page`` and ``chunk_index``
    """
    chunks = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            doc_chunks = section_chunk(full_text)
            for idx, chunk in enumerate(doc_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": "all",
                        "chunk_index": idx,
                    },
                })
    return chunks


def load_off(csv_path: Path) -> List[Dict]:
    """Load filtered Open Food Facts CSV and produce one chunk per row.

    Chunk format: ``"{product_name}: {ingredients_text} | grade:{nutrition_grade_fr}"``
    """
    df = pd.read_csv(csv_path)
    chunks = []
    for idx, row in df.iterrows():
        text = f"{row['product_name']}: {row['ingredients_text']} | grade:{row['nutrition_grade_fr']}"
        chunks.append({
            "text": text,
            "metadata": {
                "source": "off",
                "row_index": int(idx),
            },
        })
    return chunks


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> List[Dict]:
    """Generate embeddings for a list of chunk dictionaries.
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embedded = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{chunk['metadata']['source']}_{i}"
        embedded.append({
            "id": chunk_id,
            "text": chunk["text"],
            "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb),
            "metadata": chunk["metadata"],
        })
    return embedded


def main():
    # Load PDF chunks
    pdf_chunks = load_pdfs(RAW_DIR)
    print(f"Loaded {len(pdf_chunks)} PDF chunks.")

    # Load OFF chunks
    off_csv_path = RAW_DIR / "off_filtered.csv"
    off_chunks = load_off(off_csv_path)
    print(f"Loaded {len(off_chunks)} Open Food Facts chunks.")

    # Initialise model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed
    embedded_pdfs = embed_chunks(pdf_chunks, model)
    embedded_off = embed_chunks(off_chunks, model)

    # Initialise ChromaDB (persistent)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    reg_collection = client.get_or_create_collection(name="regulations")
    prod_collection = client.get_or_create_collection(name="products")

    if embedded_pdfs:
        reg_collection.add(
            ids=[c["id"] for c in embedded_pdfs],
            documents=[c["text"] for c in embedded_pdfs],
            embeddings=[c["embedding"] for c in embedded_pdfs],
            metadatas=[c["metadata"] for c in embedded_pdfs],
        )

    if embedded_off:
        prod_collection.add(
            ids=[c["id"] for c in embedded_off],
            documents=[c["text"] for c in embedded_off],
            embeddings=[c["embedding"] for c in embedded_off],
            metadatas=[c["metadata"] for c in embedded_off],
        )

    print("Embedding and storage complete.")
    print(f"Regulations collection – {len(embedded_pdfs)} chunks embedded.")
    print(f"Products collection – {len(embedded_off)} chunks embedded.")

    # Write a short JSON summary
    summary = {
        "regulations_chunks": len(embedded_pdfs),
        "products_chunks": len(embedded_off),
        "chroma_path": str(CHROMA_DB_DIR),
    }
    summary_path = REPO_ROOT / "knowledge_base" / "chroma_db" / "embedding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
