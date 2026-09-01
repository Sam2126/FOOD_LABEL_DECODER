"""
chunker.py (Data Service)

Same record-level chunking strategy explained in Exercise 2. Now living
inside the Data Service, which OWNS the knowledge base end-to-end:
raw records -> chunks -> embeddings -> vector storage.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

KB_PATH = Path(
    os.getenv("KB_PATH", str(Path(__file__).resolve().parent.parent / "knowledge_base" / "data.json"))
)


def load_knowledge_base(path: Path = KB_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_to_chunk_text(record: Dict[str, Any]) -> str:
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
    if records is None:
        records = load_knowledge_base()

    return [
        {
            "chunk_id": record["id"],
            "text": record_to_chunk_text(record),
            "metadata": record,
        }
        for record in records
    ]
