
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.agents.parser_agent import DocumentParserAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.retrieval_agent import RAGRetrievalAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.dtos import RiskLevel

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="📄",
    layout="wide",
)

RISK_COLORS = {
    "RIDICAT":   "#ffe3e3",
    "MEDIU":     "#fff3bf",
    "SCAZUT":    "#fff9c4",
    "CONFORM":   "#d3f9d8",
    "NECUNOSCUT": "#f1f3f5",
}
RISK_EMOJI = {
    "RIDICAT": "🔴",
    "MEDIU": "🟡",
    "SCAZUT": "🟢",
    "CONFORM": "✅",
    "NECUNOSCUT": "⚪",
}


def render_sidebar():
    st.sidebar.title("Legal Contract Analyzer")
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(
        "Încarcă un contract PDF",
        type=["pdf"],
        help="Suportat: contracte în limba română, format digital (nu scanat).",
    )

    st.sidebar.markdown("### Configurare retrieval")
    threshold = st.sidebar.slider(
        "Prag similaritate (threshold)",
        min_value=0.0, max_value=1.0, value=0.10, step=0.05,
        help="Chunk-urile sub acest scor sunt ignorate. Valoare mai mică = mai mult context.",
    )

    risk_alert_threshold = st.sidebar.slider(
        "Prag alertă risc RIDICAT (nr. clauze)",
        min_value=1, max_value=10, value=2, step=1,
        help="Câte clauze RIDICAT declanșează banner-ul de avertizare.",
    )

    analyze_clicked = st.sidebar.button("Analizează contractul", use_container_width=True)

    return uploaded_file, analyze_clicked, threshold, risk_alert_threshold


def render_metadata(parsed_doc):
    st.subheader("📋 Metadate document")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pagini", parsed_doc.metadata.page_count)
    col2.metric("Sectiuni", len(parsed_doc.sections))
    col3.metric("Clauze", len(parsed_doc.clauses))

    col4, col5, col6 = st.columns(3)
    col4.write(f"**Titlu:** {parsed_doc.metadata.title or 'N/A'}")
    col5.write(f"**Data semnarii:** {parsed_doc.metadata.signing_date or 'N/A'}")
    col6.write(f"**Valoare:** {parsed_doc.metadata.value or 'N/A'}")

    if parsed_doc.metadata.parties:
        st.write("**Parti contractante:**")
        for party in parsed_doc.metadata.parties:
            st.write(f"- {party.name}")


def render_clauses_table(parsed_doc, risk_map=None):
    st.subheader("Clauze extrase")
    if not parsed_doc.clauses:
        st.warning("Nu au fost extrase clauze.")
        return

    rows = []
    for clause in parsed_doc.clauses:
        risk_dto = risk_map.get(clause.id) if risk_map else None
        risk_label = risk_dto.risk_level.value if risk_dto else "—"
        emoji = RISK_EMOJI.get(risk_label, "")
        rows.append({
            "ID": clause.id,
            "Tip": clause.type.value,
            "Pagina": clause.page,
            "Risc": f"{emoji} {risk_label}" if risk_label != "—" else "—",
            "Text (preview)": clause.text[:200] + ("..." if len(clause.text) > 200 else ""),
        })

    df = pd.DataFrame(rows)

    def color_row(row):
        risk_val = row["Risc"].split()[-1] if row["Risc"] != "—" else ""
        bg = RISK_COLORS.get(risk_val, "#ffffff")
        return [
            f"background-color: {bg}; color: black;"
        ] * len(row)
    styled = df.style.apply(color_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_risk_details(parsed_doc, risk_map, context_map, recommendations):
    st.subheader("Detalii clauze riscante")

    risky = [
        c for c in parsed_doc.clauses
        if risk_map.get(c.id) and risk_map[c.id].risk_level in (RiskLevel.RIDICAT, RiskLevel.MEDIU)
    ]

    if not risky:
        st.success("Nu au fost identificate clauze cu risc ridicat sau mediu.")
        return

    rec_map = {r.clause_id: r for r in (recommendations or [])}

    for clause in risky:
        risk_dto = risk_map[clause.id]
        level = risk_dto.risk_level.value
        emoji = RISK_EMOJI.get(level, "")
        bg_color = RISK_COLORS.get(level, "#ffffff")

        with st.expander(f"{emoji} {clause.id} — {level} | {clause.type.value} | pag. {clause.page}"):
            st.markdown(
                f"<div style='background:{bg_color};padding:10px;border-radius:6px;margin-bottom:8px'>"
                f"<strong>Text original:</strong><br>{clause.text[:600]}{'...' if len(clause.text) > 600 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if risk_dto.issues:
                st.markdown("**Probleme identificate:**")
                for issue in risk_dto.issues:
                    st.write(f"• {issue}")

            if risk_dto.references:
                st.markdown("**Referinte legislative:**")
                for ref in risk_dto.references:
                    st.write(f"• {ref}")

            rec = rec_map.get(clause.id)
            if rec and rec.reformulated_text:
                st.markdown("**Reformulare propusa:**")
                st.info(rec.reformulated_text)
                if rec.explanation:
                    st.caption(f"_{rec.explanation}_")
                if rec.sources:
                    st.markdown("**Surse corpus:**")
                    for src in rec.sources[:3]:
                        st.caption(f"`{src}`")

            chunks = context_map.get(clause.id, []) if context_map else []
            if chunks:
                with st.expander(f"Context juridic recuperat ({len(chunks)} chunk-uri)", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Sursa {i} — scor {chunk.score:.3f}**")
                        st.caption(chunk.source)
                        st.write(chunk.text[:400])
                        st.divider()


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path(tempfile.gettempdir()) / "legal_contract_analyzer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)



def main():
    uploaded_file, analyze_clicked, threshold, risk_alert_threshold = render_sidebar()

    st.title("Analiza contractelor juridice")
    st.markdown("Sistem multi-agent pentru identificarea clauzelor riscante și generarea reformulărilor.")

    if analyze_clicked:
        if uploaded_file is None:
            st.error("Încarcă mai întâi un fișier PDF.")
            return

        for key in ["parsed_doc", "context_map", "risk_map", "recommendations", "report_text"]:
            st.session_state.pop(key, None)

        pdf_path = save_uploaded_file(uploaded_file)

        with st.status("Se parsează documentul...", expanded=True) as status:
            st.write("Extrag metadate, secțiuni și clauze...")
            parser = DocumentParserAgent()
            parsed_doc = parser.parse(pdf_path)
            output_name = Path(uploaded_file.name).stem
            parser.save_parsed_json(parsed_doc, f"data/{output_name}_parsed.json")
            parser.save_parser_stats(parsed_doc, f"logs/{output_name}_parser_stats.json")
            st.write(f"{len(parsed_doc.clauses)} clauze extrase din {parsed_doc.metadata.page_count} pagini.")
            st.session_state["parsed_doc"] = parsed_doc

        with st.status("Se caută context juridic relevant (RAG)...", expanded=True) as status:
            st.write(f"Caut în corpus juridic (threshold={threshold})...")
            retrieval_agent = RAGRetrievalAgent(persist_directory="vectorstore", threshold=threshold, k=5)
            context_map = {}
            for clause in parsed_doc.clauses:
                context_map[clause.id] = retrieval_agent.retrieve(clause, k=5, threshold=threshold)
            n_with_context = sum(1 for v in context_map.values() if v)
            st.write(f"Context găsit pentru {n_with_context}/{len(context_map)} clauze.")
            st.session_state["context_map"] = context_map

        with st.status("Se evaluează riscul juridic (LLM)...", expanded=True) as status:
            st.write("Evaluez fiecare clauză cu GPT-4o-mini...")
            risk_agent = RiskAssessmentAgent()
            risk_map = risk_agent.generate_outputs(
                parsed_doc=parsed_doc,
                context_map=context_map,
                output_prefix=f"data/{output_name}",
            )
            n_high = sum(1 for d in risk_map.values() if d.risk_level == RiskLevel.RIDICAT)
            n_med = sum(1 for d in risk_map.values() if d.risk_level == RiskLevel.MEDIU)
            st.write(f" {n_high} clauze RIDICAT, {n_med} clauze MEDIU identificate.")
            st.session_state["risk_map"] = risk_map

        with st.status("Se generează reformulările (LLM + self-consistency)...", expanded=True) as status:
            st.write("Generez reformulări pentru clauzele riscante...")
            rec_agent = RecommendationAgent()
            recommendations = []
            for clause in parsed_doc.clauses:
                risk_dto = risk_map.get(clause.id)
                if risk_dto:
                    rec = rec_agent.recommend(clause, risk_dto, context_map.get(clause.id, []))
                    recommendations.append(rec)

            report_text = rec_agent.generate_report(
                results=recommendations,
                risk_map=risk_map,
                output_path=f"data/{output_name}_raport.md",
            )
            st.write(f"{sum(1 for r in recommendations if r.reformulated_text)} reformulări generate.")
            st.session_state["recommendations"] = recommendations
            st.session_state["report_text"] = report_text

        st.success("Analiză completă!")

    if "parsed_doc" not in st.session_state:
        st.info("Încarcă un contract PDF și apasă 'Analizează contractul' pentru a începe.")
        return

    parsed_doc = st.session_state["parsed_doc"]
    context_map = st.session_state.get("context_map", {})
    risk_map = st.session_state.get("risk_map", {})
    recommendations = st.session_state.get("recommendations", [])
    report_text = st.session_state.get("report_text", "")

    if risk_map:
        n_high = sum(1 for d in risk_map.values() if d.risk_level == RiskLevel.RIDICAT)
        risk_alert_threshold_val = st.session_state.get("risk_alert_threshold", 2)
        if n_high >= risk_alert_threshold:
            st.warning(
                f"⚠️ **Alertă:** {n_high} clauze cu risc RIDICAT au fost identificate. "
                f"Contractul necesită revizuire urgentă de către un specialist juridic."
            )

    tab1, tab2, tab3, tab4 = st.tabs(["Document", "Risc & Recomandări", "RAG Context", "Export"])

    with tab1:
        render_metadata(parsed_doc)
        st.divider()
        render_clauses_table(parsed_doc, risk_map)

    with tab2:
        render_risk_details(parsed_doc, risk_map, context_map, recommendations)

    with tab3:
        st.subheader("Context juridic recuperat prin RAG")
        if context_map:
            rows = []
            for clause in parsed_doc.clauses:
                results = context_map.get(clause.id, [])
                rows.append({
                    "Clauza": clause.id,
                    "Tip": clause.type.value,
                    "Surse": len(results),
                    "Scor maxim": round(results[0].score, 3) if results else 0.0,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Rulați analiza pentru a vedea rezultatele RAG.")

    with tab4:
        st.subheader("Export raport")
        if report_text:
            st.download_button(
                label="⬇Descarcă raport Markdown",
                data=report_text.encode("utf-8"),
                file_name="raport_analiza_juridica.md",
                mime="text/markdown",
                use_container_width=True,
            )
            with st.expander("Previzualizare raport"):
                st.markdown(report_text[:3000] + ("..." if len(report_text) > 3000 else ""))
        else:
            st.info("Raportul va fi disponibil după finalizarea analizei.")


if __name__ == "__main__":
    main()
