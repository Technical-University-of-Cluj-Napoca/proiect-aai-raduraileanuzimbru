from typing import TypedDict

from src.agents.parser_agent import DocumentParserAgent
from src.dtos import ParsedDocumentDTO


class WorkflowState(TypedDict, total=False):
    pdf_path: str
    parsed_doc: ParsedDocumentDTO

    # Persoana 2 - completeaza
    context_map: dict
    iteration: int

    # Persoana 3 - completeaza
    risk_map: dict
    high_risk_alert: bool
    recommendations: list
    report_path: str

def parse_document_node(state: WorkflowState) -> WorkflowState:
    parser = DocumentParserAgent()
    parsed_doc = parser.parse(state["pdf_path"])

    return {
        **state,
        "parsed_doc": parsed_doc,
    }