from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.agents.parser_agent import DocumentParserAgent
from src.agents.retrieval_agent import RAGRetrievalAgent


st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="📄",
    layout="wide",
)


def render_sidebar_upload():
    st.sidebar.title("Legal Contract Analyzer")

    uploaded_file = st.sidebar.file_uploader(
        "Încarcă un contract PDF",
        type=["pdf"],
    )

    threshold = st.sidebar.slider(
        "Threshold retrieval",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
    )

    analyze_clicked = st.sidebar.button("Analizează contractul")

    return uploaded_file, analyze_clicked, threshold


def render_document_metadata(parsed_doc):
    st.subheader("Metadate document")

    col1, col2, col3 = st.columns(3)

    col1.metric("Pagini", parsed_doc.metadata.page_count)
    col2.metric("Secțiuni", len(parsed_doc.sections))
    col3.metric("Clauze", len(parsed_doc.clauses))

    st.write("**Titlu:**", parsed_doc.metadata.title or "N/A")
    st.write("**Data semnării:**", parsed_doc.metadata.signing_date or "N/A")
    st.write("**Data intrării în vigoare:**", parsed_doc.metadata.effective_date or "N/A")
    st.write("**Valoare:**", parsed_doc.metadata.value or "N/A")
    st.write("**Durată:**", parsed_doc.metadata.duration or "N/A")

    if parsed_doc.metadata.parties:
        st.write("**Părți contractante:**")
        for party in parsed_doc.metadata.parties:
            st.write(f"- {party.name}")


def render_sections(parsed_doc):
    st.subheader("Secțiuni detectate")

    if not parsed_doc.sections:
        st.info("Nu au fost detectate secțiuni.")
        return

    rows = [
        {
            "Titlu": section.title,
            "Pagina start": section.start_page,
        }
        for section in parsed_doc.sections
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_clauses_table(parsed_doc):
    st.subheader("Clauze extrase")

    if not parsed_doc.clauses:
        st.warning("Nu au fost extrase clauze.")
        return

    rows = [
        {
            "ID": clause.id,
            "Tip": clause.type.value,
            "Pagina": clause.page,
            "Secțiune": clause.section,
            "Text": clause.text[:350] + ("..." if len(clause.text) > 350 else ""),
        }
        for clause in parsed_doc.clauses
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

def render_retrieval_results(parsed_doc, context_map):
    st.subheader("Surse juridice recuperate prin RAG")

    if not context_map:
        st.info("Nu există rezultate RAG.")
        return

    rows = []

    for clause in parsed_doc.clauses:
        results = context_map.get(clause.id, [])

        best_score = results[0].score if results else 0.0

        rows.append(
            {
                "Clauza": clause.id,
                "Tip": clause.type.value,
                "Pagina": clause.page,
                "Surse recuperate": len(results),
                "Cel mai bun scor": round(best_score, 3),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Detalii surse pe clauze")

    for clause in parsed_doc.clauses[:20]:
        results = context_map.get(clause.id, [])

        with st.expander(
            f"{clause.id} | {clause.type.value} | pagina {clause.page} | {len(results)} surse"
        ):
            st.write("**Text clauză:**")
            st.write(clause.text)

            if not results:
                st.warning("Nu au fost găsite surse peste threshold.")
                continue

            for idx, result in enumerate(results, start=1):
                st.markdown(f"**Sursa {idx} — scor {result.score:.3f}**")
                st.write(result.source)
                st.write(result.text[:1200])
                st.divider()

def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path(tempfile.gettempdir()) / "legal_contract_analyzer"
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    return str(file_path)


def main():
    uploaded_file, analyze_clicked, threshold = render_sidebar_upload()

    st.title("Analiza contractelor juridice")

    if analyze_clicked:
        if uploaded_file is None:
            st.error("Încarcă mai întâi un PDF.")
            return

        pdf_path = save_uploaded_file(uploaded_file)

        with st.spinner("Se parsează documentul..."):
            parser = DocumentParserAgent()
            parsed_doc = parser.parse(pdf_path)
            output_name = Path(uploaded_file.name).stem
            parser.save_parsed_json(parsed_doc, f"data/{output_name}_parsed.json")
            parser.save_parser_stats(parsed_doc, f"logs/{output_name}_parser_stats.json")

        st.session_state["parsed_doc"] = parsed_doc

        with st.spinner("Se caută surse juridice relevante..."):
            retrieval_agent = RAGRetrievalAgent(
                persist_directory="vectorstore",
                threshold=threshold,
                k=5,
            )

            context_map = {}

            for clause in parsed_doc.clauses:
                context_map[clause.id] = retrieval_agent.retrieve(
                    clause=clause,
                    k=5,
                    threshold=threshold,
                )

        st.session_state["context_map"] = context_map
        st.session_state["retrieval_threshold"] = threshold

        st.success("Document parsat și context juridic recuperat cu succes.")

    if "parsed_doc" in st.session_state:
        parsed_doc = st.session_state["parsed_doc"]

        render_document_metadata(parsed_doc)
        render_sections(parsed_doc)
        render_clauses_table(parsed_doc)

        if "context_map" in st.session_state:
            render_retrieval_results(
                parsed_doc,
                st.session_state["context_map"],
            )


if __name__ == "__main__":
    main()