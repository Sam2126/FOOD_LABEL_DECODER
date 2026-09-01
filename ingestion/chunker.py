"""
chunker.py
==========

Turns raw knowledge-base RECORDS (one JSON object per ingredient/additive/
allergen) into small, self-contained CHUNKS of text that are good units
for embedding + retrieval.

Why chunk at all?
------------------
Embedding models work best on short, focused pieces of text. If we embedded
one giant document containing all 26 knowledge-base entries, a similarity
search for "E621" would return the whole blob - not useful as context.
By splitting into small, ingredient-level chunks, a search for "E621" can
retrieve *just* the E621 record, keeping the context sent to Code Llama
small, relevant and cheap to process.

Chunking strategy used here: RECORD-LEVEL CHUNKING.
-----------------------------------------------------
Each knowledge-base entry (already a small, self-contained unit written by
a human) becomes exactly ONE chunk. This is different from chunking a long
article by character/token count, because our source data is already
structured into short, atomic units - splitting it further would break
apart a single ingredient's information across multiple chunks, which
would hurt retrieval quality.

- Chunk size: ~40-120 words per record (naturally small, no hard limit needed)
- Chunk overlap: none needed - records are already independent units
- If in a future exercise the knowledge base grows to include long-form
  articles (e.g. a full regulatory PDF), THOSE documents would use
  character/token-based chunking with overlap. For this project's curated
  JSON knowledge base, record-level chunking is the appropriate and
  simplest reliable strategy.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "data.json"


def load_knowledge_base(path: Path = KB_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_to_chunk_text(record: Dict[str, Any]) -> str:
    """
    Flattens one knowledge-base record into a single readable text chunk.
    This is the exact text that will be embedded and later shown to
    Code Llama as retrieved context.
    """
    synonyms = ", ".join(record.get("synonyms", [])) or "None"
    e_number = record.get("e_number") or "N/A"

    return (
        f"Name: {record['name']}\n"
        f"Also known as: {synonyms}\n"
        f"E-number: {e_number}\n"
        f"Category: {record['category']}\n"
        f"Function: {record['function']}\n"
        f"Common uses: {record['common_uses']}\n"
        f"Allergen information: {record['allergen_info']}\n"
        f"Health considerations: {record['health_considerations']}\n"
        f"Source: {record['source']}"
    )


def build_chunks(records: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of chunk dicts:
        {
          "chunk_id": "...",
          "text": "...",        <- what gets embedded
          "metadata": {...}     <- original record, for display/debug
        }
    """
    if records is None:
        records = load_knowledge_base()

    chunks = []
    for record in records:
        chunks.append(
            {
                "chunk_id": record["id"],
                "text": record_to_chunk_text(record),
                "metadata": record,
            }
        )
    return chunks


if __name__ == "__main__":
    # Quick manual test: python ingestion/chunker.py
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from knowledge base.\n")
    print("Example chunk:\n")
    print(chunks[0]["text"])
