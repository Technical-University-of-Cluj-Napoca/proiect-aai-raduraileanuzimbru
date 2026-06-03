from typing import TypedDict

from src.agents.parser_agent import DocumentParserAgent
from src.agents.retrieval_agent import RAGRetrievalAgent
from src.dtos import ParsedDocumentDTO, RetrievalResultDTO


MAX_ITER = 2


class WorkflowState(TypedDict, total=False):
    pdf_path: str
    parsed_doc: ParsedDocumentDTO

    context_map: dict[str, list[RetrievalResultDTO]]
    iteration: int
    retrieval_threshold: float
    should_retry_retrieval: bool

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
        "iteration": state.get("iteration", 0),
    }


def retrieve_context_node(state: WorkflowState) -> WorkflowState:
    """
    Nod LangGraph pentru Persoana 2.
    Pentru fiecare clauza extrasa, cauta surse juridice relevante in ChromaDB.
    Rezultatul este context_map:
    {
        "clause_1": [RetrievalResultDTO, RetrievalResultDTO, ...],
        "clause_2": [...]
    }
    """
    parsed_doc = state["parsed_doc"]
    iteration = state.get("iteration", 0)

    base_threshold = state.get("retrieval_threshold", 0.10)

    threshold = max(0.05, base_threshold - iteration * 0.05)

    retrieval_agent = RAGRetrievalAgent(
        persist_directory="vectorstore",
        threshold=threshold,
        k=5,
    )

    context_map: dict[str, list[RetrievalResultDTO]] = {}

    for clause in parsed_doc.clauses:
        context_map[clause.id] = retrieval_agent.retrieve(
            clause=clause,
            k=5,
            threshold=threshold,
        )

    return {
        **state,
        "context_map": context_map,
        "retrieval_threshold": threshold,
    }


def quality_check_node(state: WorkflowState) -> WorkflowState:
    """
    Nod LangGraph pentru verificarea calitatii retrieval-ului.
    Daca prea multe clauze nu au context juridic, workflow-ul poate cere retry.
    """
    parsed_doc = state["parsed_doc"]
    context_map = state.get("context_map", {})
    iteration = state.get("iteration", 0)

    total_clauses = len(parsed_doc.clauses)

    if total_clauses == 0:
        empty_context_rate = 1.0
    else:
        empty_count = sum(
            1
            for clause in parsed_doc.clauses
            if not context_map.get(clause.id)
        )
        empty_context_rate = empty_count / total_clauses

    should_retry = empty_context_rate > 0.40 and iteration < MAX_ITER

    return {
        **state,
        "should_retry_retrieval": should_retry,
        "iteration": iteration + 1 if should_retry else iteration,
    }