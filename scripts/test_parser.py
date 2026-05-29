import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.agents.parser_agent import DocumentParserAgent

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_parser.py path/catre/contract.pdf")
        return

    pdf_path = sys.argv[1]
    pdf_name = Path(pdf_path).stem

    parser = DocumentParserAgent()
    parsed_doc = parser. parse(pdf_path)

    parsed_output = f"data/{pdf_name}_parsed.json"
    stats_output = f"logs/{pdf_name}_parser_stats.json"

    parser.save_parsed_json(parsed_doc, parsed_output)
    parser.save_parser_stats(parsed_doc, stats_output)

    print("\nParser result ")
    print(f"Titlu: {parsed_doc.metadata.title}")
    print(f"Pagini: {parsed_doc.metadata.page_count}")
    print(f"Sectiuni extrase: {len(parsed_doc.sections)}")
    print(f"Clauze extrase: {len(parsed_doc.clauses)}")
    print(f"Output JSON: {parsed_output}")
    print(f"Stats JSON: {stats_output}")

    print("\nPrimele 3 clauze ")
    for clause in parsed_doc.clauses[:3]:
        print("-" * 60)
        print(f"ID: {clause.id}")
        print(f"Tip: {clause.type.value}")
        print(f"Pagina: {clause.page}")
        print(clause.text[:1000])


if __name__ == "__main__":
    main()