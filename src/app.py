from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.agents.parser_agent import DocumentParserAgent


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

    analyze_clicked = st.sidebar.button("Analizează contractul")

    return uploaded_file, analyze_clicked


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


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path(tempfile.gettempdir()) / "legal_contract_analyzer"
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    return str(file_path)


def main():
    uploaded_file, analyze_clicked = render_sidebar_upload()

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
        st.success("Document parsat cu succes.")

    if "parsed_doc" in st.session_state:
        parsed_doc = st.session_state["parsed_doc"]

        render_document_metadata(parsed_doc)
        render_sections(parsed_doc)
        render_clauses_table(parsed_doc)


if __name__ == "__main__":
    main()