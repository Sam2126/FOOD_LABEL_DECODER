"""
rag_pipeline.py
================

Ties retrieval + prompt construction + LLM generation together:
    Context + Question -> Ollama -> Model -> Grounded Response

Orchestrates RAG generation with grounded prompt constraints and allergen transparency.
"""

import time
from typing import Optional

from app.ollama_client import ask_ollama
from rag.retriever import retrieve_for_ingredient_list, retrieve_mapped_ingredients
from rag.guardrails import validate_input_guardrails, apply_output_guardrails



def format_context(retrieved: list[dict]) -> str:
    """
    Turns retrieved chunks into a structured CONTEXT block with similarity scores.
    """
    if not retrieved:
        return "No relevant information was found in the local knowledge base for these ingredients."

    blocks = []
    for i, r in enumerate(retrieved, start=1):
        chunk_name = r.get("chunk", {}).get("metadata", {}).get("name", f"Source {i}")
        source_ref = r.get("chunk", {}).get("metadata", {}).get("source", "Curated Reference")
        text = r.get("chunk", {}).get("text", "")
        score = r.get("score", 0.0)
        blocks.append(
            f"[Source {i}: {chunk_name} (Similarity: {score:.2f}) | Citation: {source_ref}]\n{text}"
        )
    return "\n\n".join(blocks)


def format_mapped_context(mapped_items: list[dict]) -> str:
    """
    Builds a precise 1-to-1 evidence mapping for each ingredient in the user's label.
    """
    lines = []
    for item in mapped_items:
        query = item["query"]
        match = item["best_match"]
        if match:
            meta = match.get("metadata", {})
            name = meta.get("name", query)
            func = meta.get("function", "")
            allergen = meta.get("allergen_info", "Not recognized as a major allergen.")
            health = meta.get("health_considerations", "")
            lines.append(
                f"• USER INGREDIENT: \"{query}\"\n"
                f"  - Verified Knowledge Base Match: {name} (Similarity: {item['score']:.2f})\n"
                f"  - Functional Role: {func}\n"
                f"  - Allergen Status: {allergen}\n"
                f"  - Health Note: {health}"
            )
        else:
            lines.append(f"• USER INGREDIENT: \"{query}\" -> No specific record in local Codex/FDA database.")
    return "\n\n".join(lines)


def build_rag_prompt(ingredient_list: str, context: str, unmatched_ingredients: Optional[list[str]] = None) -> str:
    """
    Constructs a high-precision grounded prompt commanding a structured, consumer-grade report.
    """
    unmatched_block = ""
    if unmatched_ingredients:
        unmatched_str = ", ".join(unmatched_ingredients)
        unmatched_block = f"""
UNMATCHED INGREDIENTS (No entry in local database):
{unmatched_str}
"""

    return f"""You are a senior food scientist and consumer label specialist.
Explain ONLY the ingredients present in the USER INGREDIENT LIST below. Base all facts strictly on the VERIFIED KNOWLEDGE BASE EVIDENCE.

VERIFIED KNOWLEDGE BASE EVIDENCE (Mapped per ingredient):
{context}

USER INGREDIENT LIST:
{ingredient_list}

INSTRUCTIONS:
1. Explain EACH listed ingredient in order:
   - **[Ingredient Name]**: [What it is and its function in food]. [Allergen: Note allergen status]. [Health note: Scientific summary].
2. For items with no local record, state: "- **[Item Name]**: No specific entry in local Codex/FDA knowledge base."
3. Do NOT discuss or invent ingredients not present in the user's list.

### 🔬 Ingredient-by-Ingredient Breakdown
"""






def build_no_rag_prompt(ingredient_list: str) -> str:
    """
    Builds the baseline (Exercise 1) prompt without retrieved context.
    """
    return f"""You are a food label assistant. Explain the following ingredients to an everyday consumer:
1. What each ingredient is and its likely function in food.
2. Common allergens to watch for.
3. General health considerations.

Ingredient list:
{ingredient_list}

Organize your answer ingredient by ingredient.
"""


import re


def detect_allergens_and_dietary(
    retrieved_chunks: list[dict],
    ingredient_text: str,
    explanation: str = "",
) -> tuple[list[str], list[str], list[dict]]:
    """
    Precision extraction of food allergens, dietary certification flags,
    and direct ingredient-to-allergen mapping.
    """
    allergens = set()
    breakdown = []
    seen_pairs = set()

    # Rule dictionary: (Regex, Allergen Group, Clinical / Sensitivity Note, Severity)
    rules = [
        (
            r"\b(wheat|maida|atta|semolina|sooji|suji|durum|spelt|barley|rye|malt extract|gluten)\b",
            "Gluten / Wheat",
            "Contains gluten proteins; triggers reactions in celiac disease & wheat allergy.",
            "Critical",
        ),
        (
            r"\b(milk|dairy|butter|cheese|whey|casein|caseinate|lactose|milk solids|milk powder|cream|ghee|paneer|yogurt|curd)\b",
            "Milk / Dairy",
            "Contains dairy proteins and lactose; triggers milk allergy and lactose intolerance.",
            "Critical",
        ),
        (
            r"\b(soy|soya|soybean|soy lecithin|soya lecithin|tofu|edamame)\b",
            "Soy",
            "Recognized major legume allergen derived from soybeans.",
            "Moderate-High",
        ),
        (
            r"\b(peanut|peanuts|groundnut|groundnuts|peanut butter|peanut oil|groundnut oil|groundnut protein|peanut protein)\b",
            "Peanuts / Groundnut",
            "Contains proteins derived from peanuts/groundnuts (Arachis hypogaea). Major legume allergen capable of triggering severe anaphylaxis.",
            "Severe",
        ),

        (
            r"\b(almond|almonds|walnut|walnuts|cashew|cashews|pistachio|pistachios|hazelnut|hazelnuts|pecan|pecans|macadamia)\b",
            "Tree Nuts",
            "Tree nut botanical allergen group; distinct from peanut allergies.",
            "Severe",
        ),
        (
            r"\b(egg|eggs|egg powder|egg white|egg yolk|albumin|ovalbumin)\b",
            "Egg",
            "Contains avian albumin proteins; major pediatric & adult allergen.",
            "Critical",
        ),
        (
            r"\b(sulfur dioxide|sulphur dioxide|sulfite|sulphite|e220|e221|e222|e223|e224|e225|e226|e227|e228|ins 220|ins 221|ins 222|ins 223|ins 224)\b",
            "Sulphites",
            "Chemical preservative capable of triggering severe bronchial spasms in sulfite-sensitive individuals.",
            "Sensitivity",
        ),
        (
            r"\b(fish|anchovy|anchovies|salmon|tuna|crustacean|crustaceans|shrimp|shrimps|prawn|prawns|crab|crabs|lobster|shellfish)\b",
            "Fish / Shellfish",
            "Marine parvalbumin/tropomyosin proteins; major life-threatening allergen.",
            "Severe",
        ),
        (
            r"\b(sesame|sesame seeds|sesame oil|tahini|til)\b",
            "Sesame",
            "Recognized major food allergen (FDA FASTER Act 2023).",
            "Critical",
        ),
        (
            r"\b(tartrazine|e102|ins 102)\b",
            "Tartrazine (Sensitivity)",
            "Synthetic azo colorant associated with intolerance and histamine release.",
            "Sensitivity",
        ),
    ]

    # Tokenize input using comma/semicolon/parenthesis separators
    raw_tokens = re.split(r"[,;\n•·*\t\r]+", ingredient_text)
    for tok in raw_tokens:
        clean_tok = re.sub(r"^[~•·\*\-\d\.\s]+", "", tok).strip()
        if not clean_tok or len(clean_tok) < 2:
            continue
        clean_lower = clean_tok.lower()

        for pattern, allergen_name, note, severity in rules:
            if re.search(pattern, clean_lower):
                # Avoid classifying gluten-free flours as wheat
                if allergen_name == "Gluten / Wheat" and re.search(r"\b(rice flour|potato flour|corn flour|tapioca flour|gluten-free flour)\b", clean_lower) and not re.search(r"\b(wheat flour|wheat|maida|atta|gluten)\b", clean_lower):
                    continue
                allergens.add(allergen_name)
                pair_key = (clean_tok, allergen_name)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    breakdown.append({
                        "ingredient": clean_tok,
                        "allergen": allergen_name,
                        "severity": severity,
                        "note": note,
                    })

    # Dietary certification flags
    flags = set()

    if "Gluten / Wheat" not in allergens:
        flags.add("Gluten-Free")
    if not any(a in allergens for a in ["Milk / Dairy", "Egg", "Fish / Shellfish"]):
        flags.add("100% Vegan Friendly")
    elif "Fish / Shellfish" not in allergens:
        flags.add("Vegetarian Friendly")
    if "Milk / Dairy" not in allergens:
        flags.add("Dairy-Free")
    if not any(a in allergens for a in ["Peanuts", "Tree Nuts"]):
        flags.add("Nut-Free")

    return sorted(list(allergens)), sorted(list(flags)), breakdown


def run_rag(
    ingredient_list: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: int = 2,
    min_similarity: float = 0.30,
) -> dict:
    """
    Executes the full RAG pipeline:
      1. Run input guardrails to sanitize and reject prompt injections.
      2. Retrieve relevant chunks for each ingredient (1-to-1 mapped).
      3. Format context and prompt.
      4. Call Ollama with model of choice.
      5. Detect allergens, apply output guardrails & clinical disclaimers.
    """
    start_time = time.perf_counter()

    # 1. Input Guardrails
    is_valid, err_msg = validate_input_guardrails(ingredient_list)
    if not is_valid:
        return {
            "explanation": f"🛡️ **Guardrail Intervention:** {err_msg}",
            "mode": "guardrail_rejected",
            "processing_time_ms": int((time.perf_counter() - start_time) * 1000),
            "model_used": model or "none",
            "retrieved_chunks": [],
            "unmatched_ingredients": [],
            "context_sent_to_llm": "",
            "detected_allergens": [],
            "dietary_flags": [],
            "allergen_breakdown": [],
            "guardrails": {"applied": True, "status": "rejected", "reason": err_msg},
        }

    mapped_items, retrieved, unmatched = retrieve_mapped_ingredients(
        ingredient_list,
        min_similarity=min_similarity,
    )

    context = format_mapped_context(mapped_items)
    prompt = build_rag_prompt(ingredient_list, context, unmatched)

    # Determine mode
    if not retrieved:
        mode = "rag_no_match"
    elif unmatched:
        mode = "rag_partial"
    else:
        mode = "rag_grounded"

    explanation = ask_ollama(
        prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Format chunks
    formatted_chunks = [
        {
            "chunk_id": r["chunk"]["chunk_id"],
            "name": r["chunk"]["metadata"]["name"],
            "source": r["chunk"]["metadata"].get("source", ""),
            "score": round(r["score"], 4),
            "text": r["chunk"]["text"],
        }
        for r in retrieved
    ]

    # Detect allergens and dietary flags
    detected_allergens, dietary_flags, allergen_breakdown = detect_allergens_and_dietary(
        retrieved_chunks=formatted_chunks,
        ingredient_text=ingredient_list,
        explanation=explanation,
    )

    # 2. Output Guardrails
    guardrail_audit = apply_output_guardrails(
        raw_explanation=explanation,
        detected_allergens=detected_allergens,
        unmatched_ingredients=unmatched,
    )
    final_explanation = guardrail_audit["explanation"]

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "explanation": final_explanation,
        "mode": mode,
        "processing_time_ms": elapsed_ms,
        "model_used": model or "codellama",
        "retrieved_chunks": formatted_chunks,
        "unmatched_ingredients": unmatched,
        "context_sent_to_llm": context,
        "detected_allergens": detected_allergens,
        "dietary_flags": dietary_flags,
        "allergen_breakdown": allergen_breakdown,
        "guardrails": guardrail_audit,
    }




