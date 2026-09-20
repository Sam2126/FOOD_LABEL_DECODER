"""Dangerous combination detection and graph builder for food ingredients."""

from itertools import combinations
from typing import List, Dict

# ── Known dangerous pairings ──────────────────────────────────────────────────
DANGEROUS_COMBINATIONS: Dict[tuple, str] = {
    ("sodium benzoate", "ascorbic acid"): "forms benzene, a carcinogen",
    ("sodium benzoate", "vitamin c"): "forms benzene, a carcinogen",
    ("nitrates", "amines"): "forms nitrosamines, linked to cancer",
    ("tartrazine", "sodium benzoate"): "linked to hyperactivity in children",
    ("yellow 5", "sodium benzoate"): "linked to hyperactivity in children",
    ("tbhq", "fish oil"): "accelerates oxidation",
}


def detect_combinations(flagged_ingredients: List[str]) -> List[Dict]:
    """Check every pair of flagged ingredients against DANGEROUS_COMBINATIONS.

    Supports exact matches, aliases, and substring matches (e.g. 'Ascorbic Acid (Vitamin C)').

    Args:
        flagged_ingredients: List of ingredient name strings (any case).

    Returns:
        List of dicts with keys ``ingredients`` (list) and ``risk`` (str).
    """
    found = []
    seen_pairs = set()

    for a, b in combinations(flagged_ingredients, 2):
        la, lb = a.lower().strip(), b.lower().strip()
        for (k1, k2), risk in DANGEROUS_COMBINATIONS.items():
            if (k1 in la and k2 in lb) or (k2 in la and k1 in lb):
                pair_key = tuple(sorted([la, lb]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    found.append({"ingredients": [a, b], "risk": risk})
                break
    return found


def build_graph(flagged_ingredients: List[str], combinations_list: List[Dict]) -> Dict:
    """Build an adjacency-list graph from flagged ingredients and their dangerous pairings.

    Args:
        flagged_ingredients: List of flagged ingredient name strings.
        combinations_list: Output of :func:`detect_combinations`.

    Returns:
        Dict with:
          - ``nodes``: list of ``{"id": str, "flagged": bool}``
          - ``edges``: list of ``{"source": str, "target": str, "risk": str}``
    """
    lowered_flagged = {i.lower().strip() for i in flagged_ingredients}

    # Collect all ingredient names referenced in edges too
    all_nodes: set = set(lowered_flagged)
    for combo in combinations_list:
        for ing in combo["ingredients"]:
            all_nodes.add(ing.lower().strip())

    nodes = [
        {"id": node, "flagged": node in lowered_flagged}
        for node in sorted(all_nodes)
    ]

    edges = [
        {
            "source": combo["ingredients"][0],
            "target": combo["ingredients"][1],
            "risk": combo["risk"],
        }
        for combo in combinations_list
    ]

    return {"nodes": nodes, "edges": edges}
