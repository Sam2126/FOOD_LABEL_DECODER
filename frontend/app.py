"""Food Label Decoder — Streamlit Frontend
3-tab interface: Analyze Label | RAG Comparison | Pipeline Trace
"""
import json
import os
import requests
import streamlit as st

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Food Label Decoder",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f172a; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 8px 20px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    .flag-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .grade-a { color: #22c55e; font-weight: 700; }
    .grade-b { color: #84cc16; font-weight: 700; }
    .grade-c { color: #f59e0b; font-weight: 700; }
    .grade-d { color: #ef4444; font-weight: 700; }
    .trace-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-family: monospace;
        font-size: 14px;
    }
    .step-ok { border-left: 3px solid #22c55e; }
    .step-err { border-left: 3px solid #ef4444; }
    .step-skip { border-left: 3px solid #94a3b8; }
    .alt-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px 0 12px 0;">
    <h1 style="color:#38bdf8; margin:0; font-size:2rem;">🍱 Food Label Decoder</h1>
    <p style="color:#64748b; margin:4px 0 0 0;">AI-powered ingredient safety analysis with RAG-enhanced knowledge</p>
</div>
""", unsafe_allow_html=True)

# ── Shared input area ─────────────────────────────────────────────────────────
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload label image", type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="file_upload"
        )
        pasted_text = st.text_area(
            "Or paste ingredient list",
            placeholder="e.g. Water, Sugar, Sodium Benzoate, Tartrazine, Ascorbic Acid…",
            height=100,
            key="pasted_text",
        )
    with col2:
        product_name = st.text_input("Product name", value="Unknown Product", key="product_name")
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)


def _call_orchestrator(endpoint: str) -> dict:
    """POST to orchestrator and return JSON."""
    try:
        if uploaded_file:
            uploaded_file.seek(0)
            res = requests.post(
                f"{ORCHESTRATOR_URL}{endpoint}",
                files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                data={"product_name": product_name},
                timeout=120,
            )
        else:
            res = requests.post(
                f"{ORCHESTRATOR_URL}{endpoint}",
                data={"text": pasted_text, "product_name": product_name},
                timeout=120,
            )
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to orchestrator at {ORCHESTRATOR_URL}. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


# ── Session state ─────────────────────────────────────────────────────────────
if "result_rag" not in st.session_state:
    st.session_state.result_rag = None
if "result_no_rag" not in st.session_state:
    st.session_state.result_no_rag = None

if analyze_btn:
    if not pasted_text and not uploaded_file:
        st.warning("Please upload an image or paste ingredient text first.")
    else:
        with st.spinner("Running analysis pipeline…"):
            st.session_state.result_rag = _call_orchestrator("/process")
        with st.spinner("Running no-RAG comparison…"):
            st.session_state.result_no_rag = _call_orchestrator("/process-no-rag")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧪 Analyze Label", "📊 RAG Comparison", "🔬 Pipeline Trace"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE LABEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    result = st.session_state.result_rag

    if result is None:
        st.info("Submit an ingredient list above to see the analysis.")
    elif "error" in result:
        st.error(result["error"])
    else:
        flags = result.get("flags", {})
        if isinstance(flags, dict) and flags.get("status") == "error":
            st.error(f"Analysis error: {flags.get('message')}")
        else:
            flagged = flags.get("flagged_ingredients", []) if isinstance(flags, dict) else []
            allergens = flags.get("allergens", []) if isinstance(flags, dict) else []
            combinations = flags.get("combinations", []) if isinstance(flags, dict) else []
            summary = flags.get("summary", "") if isinstance(flags, dict) else ""
            hallucination_risk = flags.get("hallucination_risk", "unknown") if isinstance(flags, dict) else "unknown"

            # Summary banner
            risk_color = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}.get(hallucination_risk, "#64748b")
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;margin-bottom:16px;">
                <b style="color:#38bdf8;">Summary</b><br>
                <span style="color:#e2e8f0;">{summary}</span><br><br>
                <span style="color:#94a3b8;">Hallucination risk: </span>
                <span style="color:{risk_color};font-weight:600;">{hallucination_risk.upper()}</span>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            # Flagged ingredients
            with c1:
                st.markdown("#### 🚩 Flagged Ingredients")
                if flagged:
                    for f in flagged:
                        conf = f.get("confidence", 0)
                        sup = "✅ Supported by context" if f.get("supported_by_context") else "⚠️ Not in context"
                        st.markdown(f"""
                        <div class="flag-card">
                            <b style="color:#f87171;">{f.get('name','')}</b>
                            <span style="float:right;color:#94a3b8;font-size:12px;">conf: {conf:.0%}</span><br>
                            <span style="color:#cbd5e1;font-size:13px;">{f.get('reason','')}</span><br>
                            <span style="color:#64748b;font-size:11px;">{sup}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.success("No flagged ingredients found.")

            # Allergens + combinations
            with c2:
                st.markdown("#### ⚠️ Allergens")
                if allergens:
                    for a in allergens:
                        st.markdown(f"- {a}")
                else:
                    st.write("None detected.")

                st.markdown("#### ☠️ Dangerous Combinations")
                if combinations:
                    for combo in combinations:
                        ings = " + ".join(combo.get("ingredients", []))
                        st.markdown(f"""
                        <div style="background:#450a0a;border:1px solid #7f1d1d;border-radius:6px;padding:8px 12px;margin-bottom:6px;">
                            <b style="color:#fca5a5;">{ings}</b><br>
                            <span style="color:#fecaca;font-size:12px;">{combo.get('risk','')}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.write("None detected.")

            st.divider()

            # Alternatives
            st.markdown("#### 💚 Healthier Alternatives")
            alternatives = result.get("alternatives", [])
            if alternatives:
                cols = st.columns(min(len(alternatives), 3))
                for i, alt in enumerate(alternatives[:3]):
                    grade = alt.get("grade", "?")
                    grade_class = f"grade-{grade.lower()}" if grade.lower() in "abcd" else ""
                    with cols[i]:
                        st.markdown(f"""
                        <div class="alt-card">
                            <div class="{grade_class}" style="font-size:1.5rem;">{grade}</div>
                            <b style="color:#e2e8f0;">{alt.get('product_name','')}</b><br>
                            <span style="color:#64748b;font-size:12px;">{alt.get('ingredients','')[:80]}…</span>
                        </div>""", unsafe_allow_html=True)
            else:
                st.write("No alternatives found.")

            st.divider()

            # Recipe
            st.markdown("#### 🍳 Home Recipe (without flagged ingredients)")
            recipe = result.get("recipe", {})
            if recipe and isinstance(recipe, dict) and recipe.get("recipe_name"):
                st.markdown(f"**{recipe.get('recipe_name')}**")
                col_i, col_s = st.columns(2)
                with col_i:
                    st.markdown("**Ingredients:**")
                    for ing in recipe.get("ingredients", []):
                        st.markdown(f"- {ing}")
                with col_s:
                    st.markdown("**Steps:**")
                    for i, step in enumerate(recipe.get("steps", []), 1):
                        st.markdown(f"{i}. {step}")
                if recipe.get("why_healthy"):
                    st.info(f"💡 {recipe['why_healthy']}")
            else:
                st.write("No recipe generated.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    r_rag = st.session_state.result_rag
    r_no_rag = st.session_state.result_no_rag

    if r_rag is None:
        st.info("Submit an ingredient list above to compare RAG vs No-RAG.")
    else:
        col_rag, col_norag = st.columns(2)

        def _render_flags(flags_dict, label, color):
            st.markdown(f"<h4 style='color:{color};'>{label}</h4>", unsafe_allow_html=True)
            flagged = flags_dict.get("flagged_ingredients", []) if isinstance(flags_dict, dict) else []
            if flagged:
                for f in flagged:
                    conf = f.get("confidence", 0)
                    sup = "✅" if f.get("supported_by_context") else "⚠️"
                    st.markdown(f"**{f.get('name','')}** {sup} — conf: {conf:.0%}")
                    st.caption(f.get("reason", ""))
            else:
                st.write("No flagged ingredients.")
            hallucination_risk = flags_dict.get("hallucination_risk", "unknown") if isinstance(flags_dict, dict) else "unknown"
            risk_color = {"low": "green", "medium": "orange", "high": "red"}.get(hallucination_risk, "gray")
            st.markdown(f"Hallucination risk: :{risk_color}[**{hallucination_risk.upper()}**]")

        with col_rag:
            st.markdown("### 🧠 With RAG")
            _render_flags(r_rag.get("flags", {}), "Flagged (RAG)", "#38bdf8")

        with col_norag:
            st.markdown("### 🤷 Without RAG")
            _render_flags((r_no_rag or {}).get("flags", {}), "Flagged (No-RAG)", "#f59e0b")

        # Highlight differences
        st.divider()
        st.markdown("#### 🔍 Differences")
        rag_names = {f.get("name","").lower() for f in (r_rag.get("flags", {}) or {}).get("flagged_ingredients", [])}
        norag_names = {f.get("name","").lower() for f in ((r_no_rag or {}).get("flags", {}) or {}).get("flagged_ingredients", [])}
        only_rag = rag_names - norag_names
        only_norag = norag_names - rag_names

        if only_rag:
            st.success(f"Only flagged **with** RAG: {', '.join(only_rag)}")
        if only_norag:
            st.warning(f"Only flagged **without** RAG: {', '.join(only_norag)}")
        if not only_rag and not only_norag:
            st.info("Both approaches flagged the same ingredients.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE TRACE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    result = st.session_state.result_rag

    if result is None:
        st.info("Submit an ingredient list above to see the pipeline trace.")
    else:
        st.markdown("### 🔬 Pipeline Execution Log")

        pipeline_trace = result.get("pipeline_trace", [])
        for step in pipeline_trace:
            status = step.get("status", "")
            is_ok = status in ("ok", "pass", "skipped (text provided)", "skipped")
            is_err = "error" in status.lower()
            icon = "✅" if is_ok else ("❌" if is_err else "⏭️")
            css_class = "step-ok" if is_ok else ("step-err" if is_err else "step-skip")
            summary = step.get("output_summary", "")
            dur = step.get("duration_ms", 0)
            st.markdown(f"""
            <div class="trace-step {css_class}">
                {icon} <b>{step.get('service','')}</b>
                &nbsp;—&nbsp;<span style="color:#94a3b8;">{dur:.0f}ms</span>
                &nbsp;—&nbsp;<span style="color:#64748b;">{summary}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Retrieved chunks
        retrieval_results = result.get("retrieval_results", [])
        if retrieval_results:
            with st.expander(f"📚 ChromaDB Chunks Retrieved ({len(retrieval_results)})", expanded=False):
                for i, chunk in enumerate(retrieval_results, 1):
                    st.markdown(f"**{i}. [{chunk.get('collection','')}] {chunk.get('source','')}** "
                                f"— score: {chunk.get('similarity_score', 0):.3f}")
                    st.code(chunk.get("text", "")[:400], language=None)

        # Raw JSON outputs (expandable)
        with st.expander("🔩 Raw JSON — Full Pipeline Response", expanded=False):
            st.json(result)

        if st.session_state.result_no_rag:
            with st.expander("🔩 Raw JSON — No-RAG Response", expanded=False):
                st.json(st.session_state.result_no_rag)
