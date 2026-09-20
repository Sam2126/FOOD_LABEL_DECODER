"""
Multi-layer AI Guardrail Engine for Food Label Decoder.

Layer 1: Input Guardrails (Pre-LLM)
  - Scope & Domain Enforcement
  - Prompt Injection & Jailbreak Defense
  - Input Sanitization & Length Restriction
  - Information Sufficiency Check

Layer 2: Output Guardrails (Post-LLM)
  - Schema & Structure Conformance
  - Context Grounding & Hallucination Filter
  - Unsupported Severe Claims Filter
  - Appropriate Refusal Verification
  - Medical & Regulatory Disclaimer Injection
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = 4000  # characters

# Patterns that indicate out-of-scope, harmful, or jailbreak intent
OUT_OF_SCOPE_PATTERNS = [
    # General programming / code generation
    (r"\b(write|generate|create|code|implement|build|make)\b.*\b(python|java|javascript|html|css|sql|script|program|function|class|api|app)\b",
     "code_generation", "I can only analyze food labels and ingredients. I cannot write code or programs."),
    # Homework / essay writing
    (r"\b(write|essay|assignment|homework|thesis|report|article|summarize)\b.*\b(about|on|for|regarding)\b",
     "essay_writing", "I can only analyze food label ingredients for safety. I cannot write essays or reports."),
    # General knowledge / trivia (expanded) — geography, history, current events, celebrities
    (r"\b(who\s+is|who\s+was|who\s+are|who\s+invented|who\s+discovered|who\s+made)\b",
     "general_knowledge", "I am specialized in food label safety analysis. Questions about people are outside my scope."),
    (r"\b(where\s+is|where\s+are|where\s+was|where\s+can\s+i\s+find)\b",
     "general_knowledge", "I am specialized in food label safety analysis. Geographic or location questions are outside my scope."),
    (r"\b(when\s+is|when\s+was|when\s+did|when\s+will)\b",
     "general_knowledge", "I am specialized in food label safety analysis. Date or event questions are outside my scope."),
    (r"\b(what\s+is\s+the\s+(capital|population|currency|language|president|prime\s+minister|flag|history))\b",
     "general_knowledge", "I am specialized in food label safety analysis. This general knowledge question is outside my scope."),
    (r"\b(history\s+of|tell\s+me\s+about|explain\s+quantum|politics|sports|weather|stock\s+market|bitcoin|crypto|celebrity|actor|movie|song|music|exam|quiz|trivia)\b",
     "general_knowledge", "I am specialized in food label safety analysis. This query is outside my scope."),
    (r"\b(located|location|country|city|state|capital\s+of|distance\s+between|how\s+far)\b",
     "general_knowledge", "I analyze food ingredient labels only. Geographic and location queries are outside my scope."),
    # Medical prescription / clinical diagnosis
    (r"\b(prescribe|diagnose|treat|cure|medication|drug\s+dose|insulin|antibiotic|therapy|clinical\s+trial|patient|physician)\b",
     "medical_advice", "I cannot provide medical advice, diagnose conditions, or recommend medications. Please consult a qualified healthcare professional."),
    # Dangerous food adulteration or harm
    (r"\b(poison\s+someone|hide\s+contaminants?|mask\s+contamination|conceal\s+additive|adulterate|lace\s+with|tamper)\b",
     "harmful_intent", "I cannot assist with any queries related to food tampering, adulteration, or concealment of harmful substances."),
    # Explicit harm / criminal intent with food (broad)
    (r"(tamper|contaminat|adulterat|poison).{0,40}(food|product|packaging|label|detected|inspector)",
     "harmful_intent", "I cannot assist with any queries related to food tampering, harm, or criminal activities."),
]

# Prompt injection / jailbreak patterns
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+(instruction|prompt|rule|guideline)",
    r"act\s+as\s+(dan|evil|jailbreak|unrestricted|unethical)",
    # System prompt reveal - broad variant
    r"(print|reveal|show|output|expose|display)\s+(your\s+)?(system\s+prompt|instructions|prompt|internal\s+rule)",
    r"(print|reveal|show|output)\s+.{0,20}\s+(system\s+prompt|instruction)",
    r"disregard\s+(safety|guideline|rule|constraint|restriction)",
    r"disregard.{0,15}(safety|guideline|restriction|rule)",  # allows 'your' between words
    r"pretend\s+you\s+(are|have\s+no)\s+(restriction|filter|limit)",
    r"you\s+are\s+now\s+(free|unrestricted|a\s+different|dan)",
    r"jailbreak",
    r"bypass\s+(guardrail|filter|safety|restriction)",
    # Override safety settings / instructions variant
    r"override\s+(safety|system|instruction|your\s+safety)",
    r"override.{0,20}(safety|restrict|setting|guideline)",
    # Disregard / without restrictions variants
    r"without\s+(restriction|safety|filter|limit|constraint)",
    r"no\s+longer\s+restricted",
    r"not\s+bound\s+by\s+(rule|safety|instruction)",
]

# Food-relevance signal keywords
FOOD_INGREDIENT_KEYWORDS = [
    "water", "sugar", "salt", "flour", "oil", "acid", "sodium", "fat",
    "protein", "fiber", "vitamin", "mineral", "extract", "syrup", "starch",
    "lecithin", "flavor", "flavour", "colour", "color", "preservative", "antioxidant",
    "ingredients", "contains", "per 100", "mg", "emulsifier", "stabilizer",
    "sweetener", "thickener", "gum", "concentrate", "powder", "e1", "e2",
    "e3", "e4", "e5", "ins", "permitted", "added", "natural", "artificial",
    "calorie", "carbohydrate", "cholesterol", "saturated", "trans", "dietary",
    "serving", "nutrition", "nutritional", "label", "ingredient", "additive",
    "allergen", "gluten", "lactose", "soy", "wheat", "peanut", "tree nut",
    "dairy", "egg", "fish", "shellfish", "milk", "caffeine", "carmine",
    "benzoate", "sorbate", "nitrate", "nitrite", "phosphate", "citrate",
    "tartrate", "acetate", "lactate", "malate", "tbhq", "bha", "bht", "msg",
]

# WH-question words that strongly suggest a non-food conversational query
NON_FOOD_QUESTION_STARTERS = (
    "where", "who", "when", "how", "why", "what", "which", "whose", "whom",
    "tell me", "explain", "define", "describe", "list the", "name the",
    "is there", "are there", "can you tell", "do you know",
)

# Known legitimate food additives (for output grounding verification)
KNOWN_FOOD_ADDITIVES_SET = {
    "sodium benzoate", "ascorbic acid", "vitamin c", "tartrazine", "yellow 5",
    "yellow 6", "sunset yellow", "red 40", "allura red", "blue 1", "brilliant blue",
    "high fructose corn syrup", "tbhq", "bha", "bht", "msg", "monosodium glutamate",
    "potassium sorbate", "sodium metabisulfite", "hydrogenated palm oil", "palm oil",
    "titanium dioxide", "carmine", "soy lecithin", "maltodextrin", "citric acid",
    "lactic acid", "acetic acid", "phosphoric acid", "xanthan gum", "carrageenan",
    "silicon dioxide", "calcium carbonate", "sodium nitrate", "sodium nitrite",
    "disodium guanylate", "disodium inosinate", "sucralose", "aspartame", "stevia",
    "saccharin", "acesulfame", "potassium", "niacinamide", "riboflavin", "e211",
    "e330", "e621", "e102", "e110", "e129", "e133", "e319", "e120", "e171",
}

REQUIRED_OUTPUT_KEYS = ["flagged_ingredients", "allergens", "combinations", "hallucination_risk", "summary"]


# ── LAYER 1: INPUT GUARDRAILS ──────────────────────────────────────────────────

def check_input_length(text: str) -> Tuple[bool, str]:
    """Guard 1.1: Restrict excessively long inputs."""
    if len(text) > MAX_INPUT_LENGTH:
        return False, (
            f"Input exceeds maximum allowed length ({len(text)} chars vs {MAX_INPUT_LENGTH} limit). "
            "Please provide a concise food label ingredient list."
        )
    return True, ""


def check_empty_or_gibberish(text: str) -> Tuple[bool, str]:
    """Guard 1.2: Reject empty, symbol-only, or keyboard-mash inputs."""
    stripped = text.strip()
    if not stripped:
        return False, "Input is empty. Please provide an ingredient list to analyze."

    # Check for extreme symbol/number ratio (gibberish)
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 10 and alpha_count / len(stripped) < 0.3:
        return False, (
            "Input appears to be non-textual or gibberish. "
            "Please provide a readable food ingredient list."
        )

    return True, ""


def check_prompt_injection(text: str) -> Tuple[bool, str]:
    """Guard 1.3: Detect and block prompt injection / jailbreak attempts."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return False, (
                "Prompt injection or jailbreak attempt detected. "
                "This system only analyzes food ingredient lists for safety compliance."
            )
    return True, ""


def check_scope(text: str) -> Tuple[bool, str, str]:
    """Guard 1.4: Ensure input is within food label / ingredient analysis scope.

    Two-stage detection:
    1. Pattern-based: matches explicit out-of-scope topic patterns.
    2. Semantic heuristic: detects natural-language questions with zero food
       signal keywords — treats them as out-of-scope general queries.

    Returns (passed, rejection_message, violation_category).
    """
    text_lower = text.lower().strip()

    # Stage 1: Pattern-based explicit topic matching
    for pattern, category, message in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False, message, category

    # Stage 2: Heuristic — is this a natural-language question with no food content?
    food_signal_count = sum(1 for kw in FOOD_INGREDIENT_KEYWORDS if kw in text_lower)
    comma_count = text.count(",")
    word_count = len(re.findall(r"\b[a-zA-Z]{3,}\b", text_lower))

    # Detect question-like structure (ends with ? or starts with WH-word)
    is_question_form = (
        text_lower.endswith("?") or
        any(text_lower.startswith(starter) for starter in NON_FOOD_QUESTION_STARTERS)
    )

    # If it reads like a question, has multiple words, but zero food signals → out of scope
    if is_question_form and food_signal_count == 0 and word_count >= 2 and comma_count == 0:
        return (
            False,
            "This appears to be a general knowledge or conversational question with no food "
            "label content detected. I am specialized in food ingredient and safety analysis only. "
            "Please provide a food product ingredient list (e.g., 'Water, Sugar, Sodium Benzoate, ...').",
            "out_of_scope",
        )

    return True, "", ""


def check_information_sufficiency(text: str) -> Tuple[bool, str]:
    """Guard 1.5: Verify enough food-label content exists to analyze.
    Only triggers for short ambiguous inputs not already caught by scope check.
    """
    text_lower = text.lower()
    matched_keywords = sum(1 for kw in FOOD_INGREDIENT_KEYWORDS if kw in text_lower)
    comma_count = text.count(",")
    word_count = len(re.findall(r"\b[a-zA-Z]{3,}\b", text))

    # Very short — fewer than 2 meaningful words
    if word_count < 2:
        return False, (
            "Insufficient information: Input appears to contain fewer than 2 words. "
            "Please provide a complete ingredient list for analysis."
        )

    # Only 1-3 words, no food keywords, no comma-separated list structure
    if word_count <= 3 and matched_keywords == 0 and comma_count == 0:
        return False, (
            "Insufficient information: Cannot identify recognizable food ingredients or "
            "nutritional content in this input. Please provide a food label ingredient list "
            "(e.g., 'Water, Sugar, Sodium Benzoate, Citric Acid ...')."
        )

    return True, ""


def run_input_guardrails(text: str) -> Dict[str, Any]:
    """Run all input guardrail layers and return structured result."""
    checks = []

    # Guard 1.1: Length
    ok, msg = check_input_length(text)
    checks.append({"guard": "input_length", "passed": ok, "message": msg})
    if not ok:
        return _rejected(text, msg, "excessive_length", checks)

    # Guard 1.2: Empty/Gibberish
    ok, msg = check_empty_or_gibberish(text)
    checks.append({"guard": "empty_gibberish", "passed": ok, "message": msg})
    if not ok:
        return _rejected(text, msg, "empty_or_gibberish", checks)

    # Guard 1.3: Prompt Injection
    ok, msg = check_prompt_injection(text)
    checks.append({"guard": "prompt_injection", "passed": ok, "message": msg})
    if not ok:
        return _rejected(text, msg, "prompt_injection", checks)

    # Guard 1.4: Scope
    ok, msg, category = check_scope(text)
    checks.append({"guard": "scope_check", "passed": ok, "message": msg, "category": category})
    if not ok:
        return _rejected(text, msg, category, checks)

    # Guard 1.5: Information Sufficiency
    ok, msg = check_information_sufficiency(text)
    checks.append({"guard": "info_sufficiency", "passed": ok, "message": msg})
    if not ok:
        return _rejected(text, msg, "insufficient_information", checks)

    return {
        "verdict": "pass",
        "input_text": text,
        "checks": checks,
        "message": "Input passed all guardrail checks.",
    }


def _rejected(text: str, reason: str, category: str, checks: List[Dict]) -> Dict[str, Any]:
    return {
        "verdict": "reject",
        "input_text": text[:200] + "..." if len(text) > 200 else text,
        "reason": reason,
        "violation_category": category,
        "checks": checks,
    }


# ── LAYER 2: OUTPUT GUARDRAILS ─────────────────────────────────────────────────

def check_schema_conformance(output: Dict) -> Tuple[bool, str, List[str]]:
    """Guard 2.1: Validate output structure against expected schema."""
    missing = [k for k in REQUIRED_OUTPUT_KEYS if k not in output]
    if missing:
        return False, f"Output missing required keys: {missing}", missing
    if not isinstance(output.get("flagged_ingredients"), list):
        return False, "flagged_ingredients must be a list", ["flagged_ingredients"]
    if not isinstance(output.get("allergens"), list):
        return False, "allergens must be a list", ["allergens"]
    return True, "", []


def check_grounding(output: Dict, source_text: str) -> Tuple[bool, str, List[str]]:
    """Guard 2.2: Verify flagged ingredients actually appear in source text.
    Removes hallucinated additives not present in the input.
    """
    source_lower = source_text.lower()
    flagged = output.get("flagged_ingredients", [])
    grounded = []
    hallucinated = []

    for item in flagged:
        name = item.get("name", "").lower() if isinstance(item, dict) else str(item).lower()
        # Check if any word from the flagged name appears in source text
        name_words = re.findall(r"\b[a-zA-Z]{3,}\b", name)
        found_in_source = any(w in source_lower for w in name_words)
        # Also check known additives
        found_in_known = any(kw in source_lower for kw in KNOWN_FOOD_ADDITIVES_SET if kw in name)

        if found_in_source or found_in_known:
            grounded.append(item)
        else:
            hallucinated.append(item.get("name", str(item)) if isinstance(item, dict) else str(item))

    if hallucinated:
        return (
            False,
            f"Hallucinated additives removed (not found in input): {hallucinated}",
            hallucinated,
        )
    return True, "", []


def check_severe_claims(output: Dict, source_text: str) -> Tuple[bool, str, int]:
    """Guard 2.3: Count severe unverified health claims.
    Verifies that carcinogen/toxicity claims are made only for known, regulated
    substances (in KNOWN_FOOD_ADDITIVES_SET). Random/unknown chemical codes with
    severe claims are treated as unverified and flagged.
    """
    severe_keywords = ["carcinogen", "toxic", "banned", "lethal", "poison", "genotoxic", "neurotoxic"]
    unverified_severe = 0

    for item in output.get("flagged_ingredients", []):
        if not isinstance(item, dict):
            continue
        reason = item.get("reason", "").lower()
        name = item.get("name", "").lower()
        has_severe_claim = any(kw in reason for kw in severe_keywords)
        if has_severe_claim:
            # ONLY accept severe claims for additives in the known regulated set
            is_known_regulated = any(kw in name for kw in KNOWN_FOOD_ADDITIVES_SET)
            if not is_known_regulated:
                unverified_severe += 1

    return unverified_severe == 0, f"Found {unverified_severe} unverified severe claims", unverified_severe


def apply_medical_disclaimer(output: Dict) -> Dict:
    """Guard 2.4: Inject standardized medical/regulatory disclaimer."""
    disclaimer = (
        "DISCLAIMER: This analysis is generated for educational and informational purposes only "
        "based on FSSAI regulatory standards. It does not constitute medical advice, clinical "
        "diagnosis, or treatment recommendations. Consult a qualified nutritionist or healthcare "
        "professional before making dietary decisions based on this report."
    )
    output["disclaimer"] = disclaimer
    return output


def run_output_guardrails(output: Dict, source_text: str) -> Dict[str, Any]:
    """Run all output guardrail layers and return validated, corrected output."""
    checks = []
    corrections_applied = []

    # Guard 2.1: Schema conformance
    ok, msg, missing = check_schema_conformance(output)
    checks.append({"guard": "schema_conformance", "passed": ok, "message": msg, "missing_keys": missing})
    if not ok:
        # Auto-repair missing keys with safe defaults
        for key in missing:
            if key == "flagged_ingredients":
                output[key] = []
            elif key == "allergens":
                output[key] = []
            elif key == "combinations":
                output[key] = []
            elif key == "hallucination_risk":
                output[key] = "unknown"
            elif key == "summary":
                output[key] = "Analysis could not be fully completed due to format errors."
        corrections_applied.append(f"Auto-repaired missing schema keys: {missing}")

    # Guard 2.2: Grounding / hallucination filter
    ok, msg, hallucinated = check_grounding(output, source_text)
    checks.append({"guard": "grounding_check", "passed": ok, "message": msg, "hallucinated": hallucinated})
    if not ok and hallucinated:
        # Remove hallucinated items
        original_count = len(output.get("flagged_ingredients", []))
        output["flagged_ingredients"] = [
            item for item in output.get("flagged_ingredients", [])
            if item.get("name", "") not in hallucinated
        ]
        filtered_count = len(output["flagged_ingredients"])
        corrections_applied.append(
            f"Removed {original_count - filtered_count} hallucinated ingredient(s): {hallucinated}"
        )

    # Guard 2.3: Severe claims verification
    ok, msg, severe_count = check_severe_claims(output, source_text)
    checks.append({"guard": "severe_claims", "passed": ok, "message": msg, "unverified_count": severe_count})

    # Guard 2.4: Inject disclaimer
    output = apply_medical_disclaimer(output)
    checks.append({"guard": "disclaimer_injection", "passed": True, "message": "Medical disclaimer attached."})

    # Summary
    all_passed = all(c["passed"] for c in checks)
    output["output_guardrail_checks"] = checks
    output["output_guardrail_passed"] = all_passed
    output["output_corrections"] = corrections_applied

    return output


# ── Combined Evaluation ────────────────────────────────────────────────────────

def evaluate_guardrail(
    input_text: str,
    output: Optional[Dict] = None,
    mode: str = "with_guardrail",
) -> Dict[str, Any]:
    """
    Comprehensive guardrail evaluation.

    Args:
        input_text: The raw user/label input text.
        output: Optional LLM analysis output dict to run output guardrails on.
        mode: "with_guardrail" (default) or "without_guardrail" for demo.

    Returns:
        {
          "input_result": {...},       # Input guardrail outcome
          "output_result": {...},      # Output guardrail outcome (if output provided)
          "guardrail_applied": bool,   # Whether guardrail was active
          "mode": str,
        }
    """
    if mode == "without_guardrail":
        # Simulate uncontrolled mode - no guardrails applied
        return {
            "input_result": {
                "verdict": "pass",
                "message": "Guardrails DISABLED — No input validation performed.",
                "checks": [],
            },
            "output_result": output or {},
            "guardrail_applied": False,
            "mode": mode,
        }

    # With guardrail
    input_result = run_input_guardrails(input_text)

    output_result = {}
    if output and input_result.get("verdict") == "pass":
        output_result = run_output_guardrails(output, input_text)

    return {
        "input_result": input_result,
        "output_result": output_result,
        "guardrail_applied": True,
        "mode": mode,
    }
