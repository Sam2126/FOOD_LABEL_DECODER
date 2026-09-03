import difflib
import os
import sqlite3
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(
    title="Food Label Decoder - Drift Service",
    description="FastAPI service for tracking label scans, detecting ingredient drift using sqlite3, and logging changes.",
    version="1.0.0"
)


class ScanRequest(BaseModel):
    product_name: Optional[str] = ""
    ingredients: Optional[str] = ""


def get_db_path() -> str:
    """Resolve database path from environment variable or standard location."""
    env_path = os.environ.get("DB_PATH") or os.environ.get("DATABASE_PATH")
    if env_path:
        return env_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(current_dir, "../../database/food_label.db")),
        os.path.abspath(os.path.join(current_dir, "../database/food_label.db")),
        os.path.abspath("food-label-decoder/database/food_label.db"),
        os.path.abspath("database/food_label.db"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    target = os.path.abspath(os.path.join(current_dir, "../../database/food_label.db"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def init_db():
    """Ensure database schema tables are created."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                ingredients_raw TEXT,
                flagged_ingredients TEXT,
                scan_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                previous_ingredients TEXT,
                new_ingredients TEXT,
                diff TEXT,
                detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    finally:
        conn.close()


# Initialize database schema
init_db()


def compute_diff(old_text: str, new_text: str) -> str:
    """Compute unified diff between old and new ingredients."""
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile="previous",
            tofile="new",
            lineterm=""
        )
    )
    diff_result = "\n".join(diff_lines)
    if not diff_result and old_text != new_text:
        diff_result = f"- {old_text}\n+ {new_text}"
    return diff_result


@app.api_route("/health", methods=["GET", "POST"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "drift"}


@app.post("/scan")
async def scan_endpoint(request: Request, payload: Optional[ScanRequest] = None):
    """
    POST /scan
    - Accepts: {"product_name": "...", "ingredients": "..."}
    - Check if product exists in scans table
    - If yes: diff old vs new ingredients, log to drift_log, return diff
    - If no: insert new scan, return {"status": "first_scan"}
    """
    init_db()

    product_name = payload.product_name if (payload and payload.product_name) else ""
    ingredients = payload.ingredients if (payload and payload.ingredients) else ""

    # Fallback to parsing JSON or Form body if needed
    if not product_name or not ingredients:
        try:
            body = await request.json()
            if isinstance(body, dict):
                product_name = product_name or body.get("product_name", "")
                ingredients = ingredients or body.get("ingredients", "") or body.get("ingredients_raw", "")
        except Exception:
            pass

    if not product_name or not ingredients:
        try:
            form = await request.form()
            product_name = product_name or form.get("product_name", "")
            ingredients = ingredients or form.get("ingredients", "") or form.get("ingredients_raw", "")
        except Exception:
            pass

    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ingredients_raw FROM scans WHERE product_name = ? ORDER BY id DESC LIMIT 1",
            (product_name,)
        )
        row = cursor.fetchone()

        if row is not None:
            previous_ingredients = row[0] or ""
            diff = compute_diff(previous_ingredients, ingredients)
            cursor.execute(
                "INSERT INTO drift_log (product_name, previous_ingredients, new_ingredients, diff) VALUES (?, ?, ?, ?)",
                (product_name, previous_ingredients, ingredients, diff)
            )
            conn.commit()
            return {"diff": diff}
        else:
            cursor.execute(
                "INSERT INTO scans (product_name, ingredients_raw) VALUES (?, ?)",
                (product_name, ingredients)
            )
            conn.commit()
            return {"status": "first_scan"}
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
