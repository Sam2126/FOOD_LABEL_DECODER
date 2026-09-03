from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Food Label Decoder - Frontend",
    description="Frontend UI dashboard service for the Food Label Decoder application.",
    version="1.0.0"
)

@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "frontend"}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Food Label Decoder</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }
            .card { background: #1e293b; padding: 24px; border-radius: 12px; border: 1px solid #334155; max-width: 600px; margin: auto; }
            h1 { color: #38bdf8; font-size: 24px; }
            p { color: #94a3b8; line-height: 1.5; }
            .status-badge { display: inline-block; padding: 4px 10px; background: #065f46; color: #34d399; border-radius: 6px; font-weight: bold; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="card">
            <span class="status-badge">FastAPI Dashboard</span>
            <h1>Food Label Decoder</h1>
            <p>Welcome to the Food Label Decoder frontend dashboard. Connect via the Orchestrator on port 8000 to decode nutrition labels, detect allergens, and find healthier alternatives.</p>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
