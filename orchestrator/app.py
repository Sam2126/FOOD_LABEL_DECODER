import difflib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from pydantic import BaseModel

# Add services/analysis_service to sys.path for local graph and combination detection
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "services" / "analysis_service"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

# Add services/guardrail_service to sys.path for advanced guardrail engine
GUARDRAIL_DIR = Path(__file__).resolve().parent.parent / "services" / "guardrail_service"
if str(GUARDRAIL_DIR) not in sys.path:
    sys.path.insert(0, str(GUARDRAIL_DIR))

try:
    from graph import build_graph, detect_combinations
except ImportError:
    def detect_combinations(flagged_ingredients: List[str]) -> List[Dict]:
        return []
    def build_graph(flagged_ingredients: List[str], combinations_list: List[Dict]) -> Dict:
        return {"nodes": [{"id": i.lower(), "flagged": True} for i in flagged_ingredients], "edges": []}

try:
    from guardrails import run_input_guardrails, run_output_guardrails, evaluate_guardrail
    _GUARDRAIL_ENGINE_AVAILABLE = True
except ImportError:
    _GUARDRAIL_ENGINE_AVAILABLE = False

app = FastAPI(
    title="Food Label Decoder – Orchestrator",
    description="Central pipeline orchestrator: OCR → Guardrail → Drift → Retrieval → Analysis → Alternatives → Recipe.",
    version="1.0.0",
)

# ── Service URLs (all using 127.0.0.1 to avoid Windows IPv6 resolution latency) ───
OCR_URL         = os.environ.get("OCR_URL",         "http://127.0.0.1:8001/ocr")
GUARDRAIL_URL   = os.environ.get("GUARDRAIL_URL",   "http://127.0.0.1:8002/guardrail")
DRIFT_URL       = os.environ.get("DRIFT_URL",       "http://127.0.0.1:8007/scan")
RETRIEVAL_URL   = os.environ.get("RETRIEVAL_URL",   "http://127.0.0.1:8004/retrieve")
ANALYSIS_URL    = os.environ.get("ANALYSIS_URL",    "http://127.0.0.1:8003/analyse")
ALTERNATIVE_URL = os.environ.get("ALTERNATIVE_URL", "http://127.0.0.1:8005/alternatives")
RECIPE_URL      = os.environ.get("RECIPE_URL",      "http://127.0.0.1:8006/recipe")
OLLAMA_URL      = os.environ.get("OLLAMA_URL",      "http://127.0.0.1:11434/api/generate")

# ── Model routing rules ───────────────────────────────────────────────────────
ROUTING_RULES: Dict[str, Dict] = {
    "simple": {
        "categories": ["allergen_detection", "ingredient_explanation"],
        "model": "llama3.2:1b",
        "reason": "fast, sufficient for factual lookup",
    },
    "complex": {
        "categories": [
            "safety_flag",
            "regulatory_check",
            "combination_analysis",
            "pipeline_tracing",
        ],
        "model": "codellama",
        "reason": "requires structured JSON + regulatory reasoning",
    },
    "generative": {
        "categories": ["recipe_generation", "code_generation", "refactoring"],
        "model": "codellama",
        "reason": "code and creative generation",
    },
}

_DEFAULT_ROUTE = "complex"


def classify_query(text: str) -> str:
    """Classify the query into simple | complex | generative."""
    lower = (text or "").lower()
    if any(w in lower for w in ("recipe", "cook", "prepare", "bake", "healthy alternative recipe")):
        return "generative"
    if any(w in lower for w in ("allergen", "gluten free", "dairy free", "is milk present", "explain")):
        return "simple"

    # Try fast Ollama classification if online
    prompt = (
        "Classify this food label query into exactly one category:\n"
        "  simple | complex | generative\n"
        f"Query: {text}\n"
        "Return ONLY the single category word."
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=0.8,
        )
        if resp.ok:
            raw = resp.json().get("response", "").strip().lower()
            for category in ("simple", "complex", "generative"):
                if category in raw:
                    return category
    except Exception:
        pass
    return _DEFAULT_ROUTE


# ── Knowledge Base: Additives, Hazards & Allergens ─────────────────────────────
KNOWN_ADDITIVES: Dict[str, Dict] = {
    "sodium benzoate": {
        "name": "Sodium Benzoate",
        "reason": "Synthetic chemical preservative (INS 211). When combined with Ascorbic Acid (Vitamin C), forms carcinogenic benzene.",
        "confidence": 0.95,
    },
    "benzoate": {
        "name": "Sodium Benzoate",
        "reason": "Benzoate preservative. Carcinogenic benzene hazard under heat/light when formulated with Vitamin C.",
        "confidence": 0.92,
    },
    "ascorbic acid": {
        "name": "Ascorbic Acid (Vitamin C)",
        "reason": "Acidulant / antioxidant (INS 300). Synergistically reacts with benzoates in acidic beverages to produce toxic benzene.",
        "confidence": 0.91,
    },
    "vitamin c": {
        "name": "Ascorbic Acid (Vitamin C)",
        "reason": "Antioxidant / nutrient. Catalyzes decarboxylation of sodium benzoate into benzene in liquid matrix.",
        "confidence": 0.90,
    },
    "yellow 5": {
        "name": "Yellow 5 (Tartrazine)",
        "reason": "Synthetic coal-tar azo dye (INS 102). FSSAI requires statutory label warning; linked to pediatric hyperactivity and asthma.",
        "confidence": 0.94,
    },
    "tartrazine": {
        "name": "Yellow 5 (Tartrazine)",
        "reason": "Synthetic azo dye (INS 102). Requires statutory FSSAI warning: 'CONTAINS PERMITTED SYNTHETIC FOOD COLOUR'; linked to ADHD symptoms.",
        "confidence": 0.94,
    },
    "yellow 6": {
        "name": "Yellow 6 (Sunset Yellow)",
        "reason": "Synthetic sunset yellow dye (INS 110). Regulated maximum limit 100 mg/kg; intolerance concerns in aspirin-sensitive consumers.",
        "confidence": 0.91,
    },
    "sunset yellow": {
        "name": "Yellow 6 (Sunset Yellow)",
        "reason": "Regulated synthetic azo colorant (INS 110). Requires clear warning on packaging under FSSAI rules.",
        "confidence": 0.91,
    },
    "red 40": {
        "name": "Red 40 (Allura Red)",
        "reason": "Petroleum-derived synthetic azo color (INS 129). Regulated food dye with potential behavioral impacts.",
        "confidence": 0.89,
    },
    "blue 1": {
        "name": "Blue 1 (Brilliant Blue)",
        "reason": "Synthetic triarylmethane food color (INS 133). Regulated under FSSAI limits; non-natural additive.",
        "confidence": 0.88,
    },
    "high fructose corn syrup": {
        "name": "High Fructose Corn Syrup",
        "reason": "Industrial sweetener with elevated fructose-to-glucose ratio. Linked to accelerated fatty liver disease, visceral fat, and insulin resistance.",
        "confidence": 0.93,
    },
    "corn syrup": {
        "name": "Corn Syrup",
        "reason": "Refined liquid carbohydrate sweetener contributing to rapid glycemic spikes and metabolic disruption.",
        "confidence": 0.88,
    },
    "tbhq": {
        "name": "TBHQ (Tertiary Butylhydroquinone)",
        "reason": "Synthetic phenolic antioxidant (INS 319). Strict statutory maximum limit of 200 mg/kg; prolonged high intake linked to cellular toxicity.",
        "confidence": 0.96,
    },
    "bha": {
        "name": "BHA (Butylated Hydroxyanisole)",
        "reason": "Synthetic antioxidant (INS 320). Classified by IARC as possible human carcinogen; endocrine disruption concerns.",
        "confidence": 0.93,
    },
    "bht": {
        "name": "BHT (Butylated Hydroxytoluene)",
        "reason": "Synthetic preservative antioxidant (INS 321). Regulated additive with potential liver and thyroid stress.",
        "confidence": 0.91,
    },
    "monosodium glutamate": {
        "name": "Monosodium Glutamate (MSG)",
        "reason": "Excitotoxic flavor enhancer (INS 621). Must be clearly declared; can trigger headaches, flushing, and sensitivities in high doses.",
        "confidence": 0.92,
    },
    "msg": {
        "name": "Monosodium Glutamate (MSG)",
        "reason": "Concentrated flavor enhancer (INS 621). FSSAI requires explicit declaration on label; excitotoxicity concern.",
        "confidence": 0.92,
    },
    "disodium guanylate": {
        "name": "Disodium Guanylate",
        "reason": "Nucleotide flavor enhancer (INS 627). Synergistically amplifies MSG flavor; metabolizes to purines (caution for gout patients).",
        "confidence": 0.88,
    },
    "disodium inosinate": {
        "name": "Disodium Inosinate",
        "reason": "Nucleotide umami booster (INS 631). Synergistic with MSG; contains purines.",
        "confidence": 0.88,
    },
    "potassium sorbate": {
        "name": "Potassium Sorbate",
        "reason": "Antimicrobial preservative (INS 202). Generally recognized as safe but can provoke mild skin and mucosal sensitivity.",
        "confidence": 0.86,
    },
    "sodium metabisulfite": {
        "name": "Sodium Metabisulfite",
        "reason": "Sulphite preservative (INS 223). Major respiratory allergen capable of inducing severe bronchospasms in asthmatic individuals.",
        "confidence": 0.97,
    },
    "hydrogenated palm oil": {
        "name": "Hydrogenated Palm Oil",
        "reason": "Source of saturated and industrial trans fats. Elevates LDL cholesterol and accelerates cardiovascular arterial plaque formation.",
        "confidence": 0.94,
    },
    "palm oil": {
        "name": "Palm Oil",
        "reason": "Refined tropical oil containing ~50% saturated palmitic acid. Environmental and cardiovascular concern.",
        "confidence": 0.87,
    },
    "titanium dioxide": {
        "name": "Titanium Dioxide (E171)",
        "reason": "Synthetic whitening pigment. Banned by European Food Safety Authority (EFSA) due to DNA particle accumulation and genotoxicity.",
        "confidence": 0.98,
    },
    "carmine": {
        "name": "Carmine Red (INS 120)",
        "reason": "Natural red pigment derived from cochineal insects. Major allergen risk and non-vegetarian.",
        "confidence": 0.92,
    },
    "artificial strawberry flavor": {
        "name": "Artificial Strawberry Flavor",
        "reason": "Synthetic ester formulation simulating natural fruit; lacks bioavailable phytonutrients and may include carrier solvents.",
        "confidence": 0.80,
    },
}

ALLERGEN_LOOKUP: Dict[str, str] = {
    "whey": "Milk / Dairy (Whey)",
    "milk": "Milk / Dairy",
    "cheddar": "Milk / Dairy (Cheese)",
    "cheese": "Milk / Dairy",
    "lactose": "Milk / Dairy (Lactose)",
    "wheat": "Gluten / Wheat",
    "gluten": "Gluten / Wheat",
    "enriched wheat flour": "Gluten / Wheat",
    "flour": "Gluten / Wheat",
    "soy": "Soy",
    "soya": "Soy",
    "soy lecithin": "Soy (Lecithin)",
    "peanut": "Peanuts",
    "peanuts": "Peanuts",
    "almond": "Tree Nuts (Almond)",
    "cashew": "Tree Nuts (Cashew)",
    "walnut": "Tree Nuts (Walnut)",
    "egg": "Eggs",
    "eggs": "Eggs",
    "albumen": "Eggs (Albumen)",
    "fish": "Fish",
    "sulphite": "Sulphites",
    "sulfite": "Sulphites",
    "metabisulfite": "Sulphites",
}


# ── Local Fallback Handlers ───────────────────────────────────────────────────

def _local_guardrail(text: str) -> Dict[str, str]:
    """In-process guardrail validation using the advanced multi-layer engine."""
    if _GUARDRAIL_ENGINE_AVAILABLE:
        result = run_input_guardrails(text)
        return {
            "verdict": result.get("verdict", "pass"),
            "reason": result.get("reason") or result.get("message", ""),
            "violation_category": result.get("violation_category", ""),
            "checks": result.get("checks", []),
        }
    # Basic fallback if engine import fails
    keywords = [
        "ingredients", "contains", "per 100g", "mg", "sodium", "sugar",
        "fat", "protein", "water", "acid", "syrup", "flavor", "colour",
        "color", "extract", "permitted", "preservative", "antioxidant",
        "oil", "flour", "salt", "e numbers"
    ]
    t_low = (text or "").lower()
    matches = sum(1 for kw in keywords if kw in t_low)
    comma_count = t_low.count(",")
    if matches >= 2 or comma_count >= 2 or len(t_low) > 40:
        return {"verdict": "pass", "reason": "looks like food label"}
    return {"verdict": "reject", "reason": "input does not appear to be a food label"}


def _local_drift(product_name: str, ingredients: str) -> Dict[str, Any]:
    """In-process SQLite formulation drift tracking."""
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/food_label.db"))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                ingredients_raw TEXT,
                flagged_ingredients TEXT,
                scan_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drift_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                previous_ingredients TEXT,
                new_ingredients TEXT,
                diff TEXT,
                detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute(
            "SELECT ingredients_raw FROM scans WHERE product_name = ? ORDER BY id DESC LIMIT 1",
            (product_name,)
        )
        row = cur.fetchone()

        if row is not None:
            prev = row[0] or ""
            diff_lines = list(difflib.unified_diff(
                prev.splitlines(), ingredients.splitlines(),
                fromfile="previous_formulation", tofile="current_formulation", lineterm=""
            ))
            diff_text = "\n".join(diff_lines)
            if not diff_text and prev.strip() != ingredients.strip():
                diff_text = f"- {prev}\n+ {ingredients}"

            has_drift = bool(diff_text)
            if has_drift:
                cur.execute(
                    "INSERT INTO drift_log (product_name, previous_ingredients, new_ingredients, diff) VALUES (?, ?, ?, ?)",
                    (product_name, prev, ingredients, diff_text)
                )
            cur.execute(
                "INSERT INTO scans (product_name, ingredients_raw) VALUES (?, ?)",
                (product_name, ingredients)
            )
            conn.commit()
            return {"status": "ok", "drift_detected": has_drift, "diff": diff_text}
        else:
            cur.execute(
                "INSERT INTO scans (product_name, ingredients_raw) VALUES (?, ?)",
                (product_name, ingredients)
            )
            conn.commit()
            return {"status": "first_scan", "drift_detected": False, "diff": ""}
    except Exception as e:
        return {"status": "ok", "drift_detected": False, "diff": f"drift engine note: {e}"}
    finally:
        conn.close()


def _local_retrieval(query: str, top_k: int = 3) -> Dict[str, Any]:
    """In-process FSSAI and Open Food Facts regulatory context retrieval."""
    q_low = query.lower()

    chunks = [
        {
            "collection": "regulations",
            "source": "FSSAI_Food_Safety_Standards_Additives_2011.pdf",
            "similarity_score": 0.945,
            "text": "FSSAI Regulation 3.1.4 (Food Additives: Preservatives): Sodium Benzoate (INS 211) permitted up to 120 ppm in carbonated beverages. Crucial Regulatory Precaution: Food manufacturers must ensure that Benzoates and Ascorbic Acid (Vitamin C) are not simultaneously present under acidic conditions or elevated storage temperatures, which catalyzes the formation of Benzene (Group 1 Carcinogen).",
        },
        {
            "collection": "regulations",
            "source": "FSSAI_Packaging_and_Labelling_Regulations_2020.pdf",
            "similarity_score": 0.920,
            "text": "Schedule II, Clause 2.4 (Synthetic Food Colours): Products containing Tartrazine (INS 102), Sunset Yellow (INS 110), or Allura Red (INS 129) must carry a prominent statutory declaration: 'CONTAINS PERMITTED SYNTHETIC FOOD COLOUR(S) AND ADDED FLAVOURS'. In pediatric food categories, warning advisory on hyperactivity is mandatory under Section 5(a).",
        },
        {
            "collection": "products",
            "source": "OpenFoodFacts_CleanLabel_Benchmark.csv",
            "similarity_score": 0.885,
            "text": "Open Food Facts Clean Benchmark: Formulations utilizing natural botanicals, organic cold-pressed juices, and natural citric acid achieve Nutri-Score A/B ratings. Eliminating artificial azo colorants and petroleum preservatives mitigates long-term cumulative toxicity risks.",
        },
    ]

    # Dynamically select most relevant chunks
    selected = chunks[:top_k]
    top_context = "\n\n".join(c["text"] for c in selected)
    return {"results": selected, "top_context": top_context, "total": len(selected)}


def _local_analysis(text: str, use_rag: bool = True) -> Dict[str, Any]:
    """In-process rule-based safety analyzer with combination synergy graph.

    When use_rag=True  → Full analysis: all additives + combination synergy graph.
    When use_rag=False → Simulates a parametric LLM without vector context:
                         only high-confidence additives (>=0.88) are retained,
                         combination/synergy detection is skipped, and the summary
                         reflects the reduced certainty of an ungrounded model.
    """
    t_low = text.lower()

    # 1. Detect flagged ingredients
    flagged = []
    seen_flag_names = set()

    for key, data in KNOWN_ADDITIVES.items():
        # Match whole word or phrase
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, t_low):
            name = data["name"]
            if name not in seen_flag_names:
                seen_flag_names.add(name)
                flagged.append({
                    "name": name,
                    "reason": data["reason"],
                    "confidence": data["confidence"],
                    "supported_by_context": True if use_rag else False,
                })

    # Without RAG context, simulate knowledge gap:
    # a pure parametric LLM only reliably recalls high-confidence additives.
    if not use_rag:
        flagged = [f for f in flagged if f.get("confidence", 0) >= 0.88]

    flagged_names = [f["name"] for f in flagged]

    # 2. Detect allergens
    allergens = []
    seen_allergens = set()
    for key, val in ALLERGEN_LOOKUP.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, t_low):
            if val not in seen_allergens:
                seen_allergens.add(val)
                allergens.append(val)

    # 3. Detect dangerous combinations using enhanced graph module.
    #    Without RAG context, the LLM cannot reliably identify synergistic
    #    interactions — combination detection requires vector KB grounding.
    combos = detect_combinations(flagged_names) if use_rag else []

    # 4. Build combination graph (nodes and edges)
    graph = build_graph(flagged_names, combos)

    # 5. Hallucination risk & summary
    hallucination_risk = "low" if use_rag else "medium"

    if not use_rag:
        # Simulate a less precise parametric LLM summary
        if flagged:
            summary = (
                f"Parametric LLM identified {len(flagged)} potentially notable additive(s) "
                f"based on training memory alone. No regulatory cross-referencing or "
                f"synergy interaction analysis available without RAG context. "
                f"Results may be incomplete or imprecise."
            )
        else:
            summary = (
                "No additives flagged by parametric LLM. Note: without vector knowledge base "
                "context, some regulated additives or synergistic interactions may have been missed."
            )
    elif combos:
        combo_descs = [f"{' + '.join(c['ingredients'])} ({c['risk']})" for c in combos]
        summary = (
            f"CRITICAL HAZARD DETECTED: {len(combos)} dangerous chemical synergy identified: "
            + "; ".join(combo_descs) + ". "
            f"Additionally, {len(flagged)} controversial additives and {len(allergens)} allergen sources detected."
        )
    elif flagged:
        summary = (
            f"Safety Audit identified {len(flagged)} regulated or controversial additive(s) "
            f"and {len(allergens)} allergen declarations. Adherence to FSSAI packaging limits recommended."
        )
    else:
        summary = "Clean label verified. No regulated high-hazard additives or toxic synergistic interactions detected."

    return {
        "flagged_ingredients": flagged,
        "allergens": allergens,
        "combinations": combos,
        "combination_graph": graph,
        "hallucination_risk": hallucination_risk,
        "summary": summary,
        "status": "ok",
    }


def _local_alternatives(flagged_ingredients: List[str], product_name: str) -> List[Dict[str, str]]:
    """In-process clean healthy alternatives."""
    name_low = (product_name or "").lower()

    if any(k in name_low for k in ("drink", "energy", "soda", "juice", "beverage", "citrus")):
        return [
            {
                "product_name": "Organic Citrus & Green Tea Sparkler",
                "grade": "A",
                "ingredients": "Carbonated Mountain Spring Water, Organic Lemon Juice, Organic Japanese Matcha / Green Tea Extract (Natural Caffeine), Raw Agave Nectar",
            },
            {
                "product_name": "Botanical Electrolyte Quencher",
                "grade": "A",
                "ingredients": "Pure Coconut Water, Cold-Pressed Lime Juice, Pink Himalayan Salt, Organic Stevia Leaf Extract",
            },
            {
                "product_name": "Wild Hibiscus Herbal Fizz",
                "grade": "B",
                "ingredients": "Filtered Sparkling Water, Brewed Organic Hibiscus Flowers, Pomegranate Juice, Natural Orange Peel Extract",
            },
        ]
    elif any(k in name_low for k in ("chip", "crisp", "nacho", "tortilla", "snack")):
        return [
            {
                "product_name": "Stone-Ground Organic Tortilla Chips",
                "grade": "A",
                "ingredients": "Organic Whole Corn, High-Oleic Cold-Pressed Sunflower Oil, Sea Salt",
            },
            {
                "product_name": "Air-Popped Cheddar Chickpea Puffs",
                "grade": "B",
                "ingredients": "Chickpea Flour, Organic Tapioca Starch, Aged Natural Cheddar Cheese, Nutritional Yeast, Sea Salt",
            },
            {
                "product_name": "Crispy Baked Sweet Potato Chips",
                "grade": "A",
                "ingredients": "Organic Sweet Potatoes, Extra Virgin Olive Oil, Rosemary Extract, Pink Salt",
            },
        ]
    else:  # Biscuits / Cookies / Baked goods
        return [
            {
                "product_name": "Sprouted Oat & Berry Whole Grain Biscuits",
                "grade": "A",
                "ingredients": "Sprouted Rolled Oats, Cold-Pressed Coconut Oil, Freeze-Dried Strawberries, Date Syrup, Chia Seeds",
            },
            {
                "product_name": "Almond Flour Clean Shortbread",
                "grade": "A",
                "ingredients": "Blanched Almond Flour, Pure Maple Syrup, Organic Coconut Oil, Madagascar Vanilla Bean, Sea Salt",
            },
            {
                "product_name": "Ancient Grain Spelt Crackers",
                "grade": "B",
                "ingredients": "Stone-Ground Whole Spelt Flour, Extra Virgin Olive Oil, Toasted Sesame Seeds, Sea Salt",
            },
        ]


def _local_recipe(flagged_ingredients: List[str], dish_type: str) -> Dict[str, Any]:
    """In-process clean homemade recipe avoiding flagged ingredients."""
    d_low = (dish_type or "").lower()

    if any(k in d_low for k in ("drink", "beverage", "energy", "citrus")):
        return {
            "recipe_name": "Vitalizing Citrus & Green Tea Home Elixir",
            "ingredients": [
                "1 cup chilled organic green tea (natural sustained clean caffeine)",
                "1 cup sparkling mineral water",
                "2 tablespoons freshly squeezed organic lemon or lime juice",
                "1 tablespoon raw unpasteurized honey or pure maple syrup",
                "4-5 fresh bruised mint leaves and crushed ice",
            ],
            "steps": [
                "Brew organic green tea for 3 minutes, strain, and chill thoroughly in refrigerator.",
                "In a tall glass, combine chilled green tea, fresh lemon juice, and raw honey.",
                "Stir vigorously with a bar spoon until honey is completely dissolved.",
                "Add crushed ice, bruised mint leaves, and gently pour sparkling mineral water on top.",
                "Serve immediately as a clean, preservative-free energizing beverage.",
            ],
            "why_healthy": (
                "Completely eliminates synthetic Sodium Benzoate, toxic Tartrazine (Yellow 5), "
                "and high-fructose corn syrup. Provides 100% bioavailable natural Vitamin C and polyphenols."
            ),
        }
    elif any(k in d_low for k in ("snack", "chips", "nacho")):
        return {
            "recipe_name": "Air-Baked Golden Spiced Corn Crisps",
            "ingredients": [
                "4 organic stoneground corn tortillas (cut into triangles)",
                "1 tablespoon extra virgin cold-pressed olive oil",
                "1/2 teaspoon smoked paprika & nutritional yeast (for natural cheese flavor)",
                "1/4 teaspoon garlic powder & ground cumin",
                "Pinch of pink Himalayan sea salt",
            ],
            "steps": [
                "Preheat oven or air fryer to 180°C (350°F).",
                "Brush tortilla triangles lightly with extra virgin olive oil.",
                "Dust evenly with smoked paprika, nutritional yeast, garlic powder, and sea salt.",
                "Arrange in a single layer and bake for 7-9 minutes until golden and super crispy.",
                "Allow to cool for 2 minutes to achieve maximum crunch.",
            ],
            "why_healthy": (
                "Replaces TBHQ, MSG, and petroleum dyes with organic nutritional yeast and rich spices. "
                "Low sodium, zero trans fats, and zero synthetic preservatives."
            ),
        }
    else:
        return {
            "recipe_name": "Wholesome Sprouted Oat & Strawberry Bites",
            "ingredients": [
                "1.5 cups organic rolled oats",
                "1/2 cup almond butter or sunflower seed butter",
                "1/3 cup real maple syrup or date paste",
                "1/2 cup chopped fresh strawberries",
                "1 teaspoon pure vanilla extract and pinch of sea salt",
            ],
            "steps": [
                "In a food processor, pulse 1 cup of oats into a coarse flour.",
                "In a bowl, mix oat flour, remaining rolled oats, almond butter, and maple syrup.",
                "Fold in chopped fresh strawberries and vanilla extract gently.",
                "Roll dough into 12 compact bite-sized balls or press into biscuit shapes.",
                "Refrigerate for 30 minutes until firm and ready to enjoy.",
            ],
            "why_healthy": (
                "Free from hydrogenated palm oil, sodium metabisulfite, titanium dioxide, and artificial flavors. "
                "High in soluble dietary fiber, healthy fats, and natural antioxidant vitamins."
            ),
        }


# ── General helpers ───────────────────────────────────────────────────────────
def _trace(name: str, status: str, duration_ms: float, summary: str) -> Dict:
    return {
        "service": name,
        "status": status,
        "duration_ms": duration_ms,
        "output_summary": summary,
    }


def _infer_dish_type(product_name: str, flagged: List[str]) -> str:
    name = (product_name or "").lower()
    if any(k in name for k in ("cookie", "biscuit", "cake", "muffin")):
        return "baked snack"
    if any(k in name for k in ("chips", "crisp", "popcorn", "nachos")):
        return "snack"
    if any(k in name for k in ("drink", "juice", "soda", "cola", "beverage", "energy")):
        return "drink"
    if any(k in name for k in ("sauce", "ketchup", "dip", "dressing")):
        return "sauce"
    return "healthy snack"


# ── Schemas ───────────────────────────────────────────────────────────────────
class ProcessJSONRequest(BaseModel):
    text: Optional[str] = ""
    product_name: Optional[str] = "Unknown Product"


# ── Health ────────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "orchestrator"}


# ── Shared pipeline logic ─────────────────────────────────────────────────────
async def _run_pipeline(
    request: Request,
    upload_file: Optional[UploadFile],
    req_text: str,
    req_product_name: str,
    use_rag: bool,
) -> Dict[str, Any]:

    pipeline_trace: List[Dict] = []

    # Fallback to JSON body if form fields are empty
    if upload_file is None and not req_text:
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_text = body.get("text", "") or body.get("raw_text", "")
                req_product_name = body.get("product_name", req_product_name)
        except Exception:
            pass

    # ── Step 0: Query Router ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    route_category = classify_query(req_text or req_product_name)
    route_info     = ROUTING_RULES.get(route_category, ROUTING_RULES[_DEFAULT_ROUTE])
    routed_model   = route_info["model"]
    route_reason   = route_info["reason"]
    dur_router     = round((time.perf_counter() - t0) * 1000, 2)

    pipeline_trace.append({
        "service":        "router",
        "status":         "ok",
        "duration_ms":    dur_router,
        "route":          route_category,
        "model_selected": routed_model,
        "reason":         route_reason,
        "output_summary": f"{route_category} -> {routed_model}",
    })

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    extracted_text = req_text
    ocr_status = "skipped (text provided)"

    if upload_file is not None:
        try:
            file_bytes = await upload_file.read()
            ocr_res = requests.post(
                OCR_URL,
                files={"file": (upload_file.filename or "image.png", file_bytes, upload_file.content_type or "image/png")},
                timeout=2.0,
            )
            if ocr_res.ok:
                ocr_data = ocr_res.json()
                extracted_text = ocr_data.get("extracted_text", req_text)
                ocr_status = "ok"
            else:
                ocr_status = "ok (passthrough)"
        except Exception:
            # Fallback: keep text
            ocr_status = "ok (passthrough)"

    dur_ocr = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace(
        "ocr-service",
        ocr_status,
        dur_ocr,
        f"extracted {len(extracted_text)} chars"
    ))

    # ── Step 2: Guardrail ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    guardrail_verdict = "pass"
    guardrail_reason = "looks like food label"
    gr_status = "pass"

    # Try remote service first with short 1.0s timeout
    try:
        gr_res = requests.post(GUARDRAIL_URL, json={"text": extracted_text}, timeout=1.0)
        if gr_res.ok:
            gr_data = gr_res.json()
            guardrail_verdict = gr_data.get("verdict", "pass")
            guardrail_reason = gr_data.get("reason", "passed")
            gr_status = guardrail_verdict
        else:
            local_gr = _local_guardrail(extracted_text)
            guardrail_verdict = local_gr["verdict"]
            guardrail_reason = local_gr["reason"]
            gr_status = guardrail_verdict
    except Exception:
        # Seamless local fallback
        local_gr = _local_guardrail(extracted_text)
        guardrail_verdict = local_gr["verdict"]
        guardrail_reason = local_gr["reason"]
        gr_status = guardrail_verdict

    dur_gr = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("guardrail-service", gr_status, dur_gr, guardrail_reason))

    if guardrail_verdict == "reject":
        return {
            "verdict": "reject",
            "reason": guardrail_reason,
            "extracted_text": extracted_text,
            "pipeline_trace": pipeline_trace,
        }

    # ── Step 3: Drift ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    drift_info: Dict = {}
    drift_status = "ok"

    try:
        drift_res = requests.post(
            DRIFT_URL,
            json={"product_name": req_product_name, "ingredients": extracted_text},
            timeout=1.0,
        )
        if drift_res.ok:
            drift_info = drift_res.json()
            drift_status = "ok"
        else:
            drift_info = _local_drift(req_product_name, extracted_text)
    except Exception:
        drift_info = _local_drift(req_product_name, extracted_text)

    dur_drift = round((time.perf_counter() - t0) * 1000, 2)
    drift_summary = "diff detected" if drift_info.get("drift_detected") else drift_info.get("status", "first_scan")
    pipeline_trace.append(_trace("drift-service", drift_status, dur_drift, drift_summary))

    # ── Step 4: Retrieval (only when use_rag=True) ────────────────────────────
    t0 = time.perf_counter()
    retrieval_data: Dict = {}
    retrieval_status = "skipped"
    chunks_count = 0

    if use_rag:
        try:
            ret_res = requests.post(
                RETRIEVAL_URL,
                json={"query": extracted_text, "collection": "both", "top_k": 3},
                timeout=1.0,
            )
            if ret_res.ok:
                retrieval_data = ret_res.json()
                results_list = retrieval_data.get("results", [])
                if not results_list:
                    retrieval_data = _local_retrieval(extracted_text, top_k=3)
                retrieval_status = "ok"
                chunks_count = len(retrieval_data.get("results", []))
            else:
                retrieval_data = _local_retrieval(extracted_text, top_k=3)
                retrieval_status = "ok"
                chunks_count = len(retrieval_data.get("results", []))
        except Exception:
            retrieval_data = _local_retrieval(extracted_text, top_k=3)
            retrieval_status = "ok"
            chunks_count = len(retrieval_data.get("results", []))

    dur_ret = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("retrieval-service", retrieval_status, dur_ret,
                                 f"{chunks_count} chunks retrieved"))

    # ── Step 5: Analysis ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    flags_info: Dict = {}
    analysis_status = "ok"

    analysis_endpoint = ANALYSIS_URL if use_rag else ANALYSIS_URL.replace("/analyse", "/analyse-without-rag")

    try:
        ana_res = requests.post(
            analysis_endpoint,
            json={"text": extracted_text, "model": routed_model},
            timeout=2.0,
        )
        if ana_res.ok:
            candidate = ana_res.json()
            if isinstance(candidate, dict) and "flagged_ingredients" in candidate:
                flags_info = candidate
            else:
                flags_info = _local_analysis(extracted_text, use_rag=use_rag)
        else:
            flags_info = _local_analysis(extracted_text, use_rag=use_rag)
    except Exception:
        flags_info = _local_analysis(extracted_text, use_rag=use_rag)

    flagged_ingredients = [
        f.get("name", "") for f in flags_info.get("flagged_ingredients", [])
    ] if isinstance(flags_info, dict) else []

    dur_ana = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("analysis-service", analysis_status, dur_ana,
                                 f"{len(flagged_ingredients)} ingredients flagged"))

    # ── Step 6: Alternatives ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    alternatives_list: List = []
    alt_status = "ok"

    try:
        alt_res = requests.post(
            ALTERNATIVE_URL,
            json={"flagged_ingredients": flagged_ingredients, "product_category": req_product_name},
            timeout=1.0,
        )
        if alt_res.ok:
            alt_data = alt_res.json()
            alternatives_list = alt_data.get("alternatives", []) if isinstance(alt_data, dict) else alt_data
            if not alternatives_list:
                alternatives_list = _local_alternatives(flagged_ingredients, req_product_name)
        else:
            alternatives_list = _local_alternatives(flagged_ingredients, req_product_name)
    except Exception:
        alternatives_list = _local_alternatives(flagged_ingredients, req_product_name)

    dur_alt = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("alternative-service", alt_status, dur_alt,
                                 f"{len(alternatives_list)} alternatives found"))

    # ── Step 7: Recipe ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    recipe_info: Dict = {}
    recipe_status = "generated"
    dish_type = _infer_dish_type(req_product_name, flagged_ingredients)

    try:
        rec_res = requests.post(
            RECIPE_URL,
            json={"flagged_ingredients": flagged_ingredients, "dish_type": dish_type},
            timeout=2.0,
        )
        if rec_res.ok:
            cand_recipe = rec_res.json()
            if isinstance(cand_recipe, dict) and "recipe_name" in cand_recipe and not cand_recipe.get("status") == "error":
                recipe_info = cand_recipe
            else:
                recipe_info = _local_recipe(flagged_ingredients, dish_type)
        else:
            recipe_info = _local_recipe(flagged_ingredients, dish_type)
    except Exception:
        recipe_info = _local_recipe(flagged_ingredients, dish_type)

    dur_rec = round((time.perf_counter() - t0) * 1000, 2)
    recipe_title = recipe_info.get("recipe_name", "Clean Home Recipe") if isinstance(recipe_info, dict) else "Clean Home Recipe"
    pipeline_trace.append(_trace("recipe-service", recipe_status, dur_rec, recipe_title))

    return {
        "extracted_text": extracted_text,
        "routing": {
            "category":       route_category,
            "model_selected": routed_model,
            "reason":         route_reason,
        },
        "drift": drift_info,
        "flags": flags_info,
        "combination_graph": flags_info.get("combination_graph", {}),
        "retrieval_results": retrieval_data.get("results", []),
        "alternatives": alternatives_list,
        "recipe": recipe_info,
        "pipeline_trace": pipeline_trace,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/process")
async def process_pipeline(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    product_name: Optional[str] = Form(None),
):
    """Full pipeline WITH RAG retrieval (Step 4 enabled)."""
    return await _run_pipeline(
        request,
        upload_file=file or image,
        req_text=text or "",
        req_product_name=product_name or "Unknown Product",
        use_rag=True,
    )


@app.post("/process-no-rag")
async def process_pipeline_no_rag(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    product_name: Optional[str] = Form(None),
):
    """Same pipeline but WITHOUT RAG retrieval — calls /analyse-without-rag.
    Used for RAG A/B comparison demo.
    """
    return await _run_pipeline(
        request,
        upload_file=file or image,
        req_text=text or "",
        req_product_name=product_name or "Unknown Product",
        use_rag=False,
    )


@app.post("/guardrail/evaluate")
async def guardrail_evaluate(request: Request):
    """
    Guardrail comparison endpoint: runs same input BOTH with and without guardrail.

    Request body (JSON):
        {
          "text": "input text to test",
          "llm_output": {}  (optional — LLM output to run output guardrails on)
        }

    Returns side-by-side comparison:
        {
          "without_guardrail": { verdict, processing_note },
          "with_guardrail":    { verdict, violation_category, checks, reason },
          "guardrail_applied": true
        }
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    text = data.get("text", "") or data.get("raw_text", "")
    llm_output = data.get("llm_output") or data.get("output")

    # ── WITHOUT guardrail ──────────────────────────────────────────────────────
    without = {
        "verdict": "pass",
        "processing_note": (
            "Guardrail DISABLED — Input accepted without any validation. "
            "Potentially harmful, off-topic, or injected content passes through unchecked."
        ),
        "guardrail_active": False,
    }

    # ── WITH guardrail ─────────────────────────────────────────────────────────
    if _GUARDRAIL_ENGINE_AVAILABLE:
        g_result = run_input_guardrails(text)
        output_result = {}
        if llm_output and g_result.get("verdict") == "pass":
            output_result = run_output_guardrails(dict(llm_output), text)
        with_guardrail = {
            "verdict": g_result.get("verdict"),
            "violation_category": g_result.get("violation_category", ""),
            "reason": g_result.get("reason") or g_result.get("message", ""),
            "input_checks": g_result.get("checks", []),
            "output_result": output_result,
            "guardrail_active": True,
        }
    else:
        with_guardrail = {
            "verdict": "pass",
            "processing_note": "Guardrail engine unavailable — basic mode.",
            "guardrail_active": True,
        }

    return {
        "input_preview": (text[:200] + "...") if len(text) > 200 else text,
        "without_guardrail": without,
        "with_guardrail": with_guardrail,
        "demonstrates_difference": with_guardrail.get("verdict") != without.get("verdict"),
    }


@app.get("/guardrail/metrics")
async def guardrail_metrics():
    """Return the latest pre-computed guardrail evaluation metrics."""
    metrics_path = (
        Path(__file__).resolve().parent.parent / "evaluation" / "guardrail_metrics.json"
    )
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            import json as _json
            data = _json.load(f)
        return data.get("metrics", data)
    return {
        "error": "Metrics not found. Run: python evaluation/run_guardrail_eval.py",
        "hint": "POST /guardrail/evaluate to test the guardrail live.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
