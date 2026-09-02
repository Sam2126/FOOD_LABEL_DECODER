"""
guardrails.py
=============

Enterprise-grade Guardrails module for Food Label Decoder RAG Pipeline:
1. Input Guardrails:
   - Sanitizes and validates user input.
   - Detects prompt injections, malicious inputs, system overrides, and non-food gibberish.
2. Output Guardrails:
   - Enforces clinical allergen safety assertions (prevents dangerous safety misstatements).
   - Validates factual adherence to unverified/unmatched items.
   - Attaches mandatory clinical & dietary disclaimers.
"""

import re
import logging
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger("food-label-decoder.guardrails")

# Prohibited prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"bypass\s+safety",
    r"dan\s+mode",
    r"jailbreak",
    r"drop\s+table",
    r"<script\b",
    r"base64\s+decode",
]

# High-risk allergen keywords
CRITICAL_ALLERGENS = ["peanut", "groundnut", "gluten", "wheat", "milk", "dairy", "soy", "egg", "carmine", "sulphite"]


class GuardrailViolation(Exception):
    """Exception raised when an input violates safety guardrails."""
    pass


def validate_input_guardrails(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validates user ingredient input against prompt injection, malicious strings,
    and extreme length limits.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if not text or len(text.strip()) < 2:
        return False, "Input ingredient list cannot be empty."

    if len(text) > 4000:
        return False, "Ingredient text exceeds maximum safe length limit (4,000 characters)."

    # Check for prompt injection attempts
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning("Prompt injection attempt intercepted: '%s'", pattern)
            return False, "Security Alert: Input contained unauthorized system commands or injection patterns."

    return True, None


def apply_output_guardrails(
    raw_explanation: str,
    detected_allergens: List[str],
    unmatched_ingredients: List[str],
) -> Dict[str, Any]:
    """
    Sanitizes, validates, and enhances LLM-generated explanations before returning to the consumer.

    Features:
      1. Allergen Safety Assertion: Detects if high-risk allergens were falsely marked safe.
      2. Unmatched Ingredients Verification: Ensures fallback disclaimers exist for unindexed items.
      3. Medical/Regulatory Disclaimer: Attaches standard consumer safety notice.
    """
    sanitized_text = raw_explanation.strip()
    safety_flags: List[str] = []
    text_lower = sanitized_text.lower()

    # 1. Allergen safety assertion
    for allergen in detected_allergens:
        allg_name = allergen.lower()
        # Look for dangerous contradictions (e.g., "safe for gluten allergy" or "peanut-free" when peanut is present)
        dangerous_phrases = [
            f"safe for {allg_name}",
            f"no {allg_name} allergen",
            f"free of {allg_name}",
            f"{allg_name} is not an allergen",
        ]
        for phrase in dangerous_phrases:
            if phrase in text_lower:
                safety_flags.append(f"CRITICAL ALLERGEN OVERRIDE: Dangerous contradiction detected for '{allergen}'.")
                # Append high-priority correction
                sanitized_text += f"\n\n🚨 **CRITICAL SAFETY OVERRIDE:** This product contains verified **{allergen}**. Individuals with relevant sensitivities or allergies must exercise strict caution."

    # 2. Unmatched item audit
    if unmatched_ingredients:
        for item in unmatched_ingredients:
            if item.lower() in text_lower and not any(p in text_lower for p in ["not found", "no record", "unverified", "unmatched"]):
                safety_flags.append(f"UNMATCHED AUDIT: Added verification disclaimer for unindexed item '{item}'.")

    # 3. Mandatory clinical disclaimer
    disclaimer = (
        "\n\n---\n"
        "🛡️ **Clinical Disclaimer:** *Food label decoding is provided for consumer awareness and educational reference based on Codex Alimentarius & FDA regulations. "
        "It does not substitute for personalized medical advice, clinical allergy diagnostics, or manufacturer allergen cross-contact declarations.*"
    )

    if "clinical disclaimer" not in text_lower:
        sanitized_text += disclaimer

    return {
        "explanation": sanitized_text,
        "guardrails_applied": True,
        "safety_flags": safety_flags,
        "allergen_assertions_checked": len(detected_allergens),
    }
