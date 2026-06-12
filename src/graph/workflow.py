import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from src.agents.parser_agent import DocumentParserAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.retrieval_agent import RAGRetrievalAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.dtos import ParsedDocumentDTO, RecommendationDTO, RetrievalResultDTO, RiskAssessmentDTO, RiskLevel

logger = logging.getLogger(__name__)

MAX_ITER = 2
HIGH_RISK_THRESHOLD = 2
UNKNOWN_RATE_THRESHOLD = 0.40


class WorkflowState(TypedDict, total=False):

    pdf_path: str

    parsed_doc: ParsedDocumentDTO

    context_map: dict[str, list[RetrievalResultDTO]]
    retrieval_threshold: float
    iteration: int

    should_retry_retrieval: bool

    risk_map: dict[str, RiskAssessmentDTO]

    high_risk_alert: bool

    recommendations: list[RecommendationDTO]

    report_path: str
    report_text: str



def parse_document_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    logger.info("[parse_document] start")

    parser = DocumentParserAgent()
    parsed_doc = parser.parse(state["pdf_path"])

    _log_node("parse_document", t0, extra={
        "sections": len(parsed_doc.sections),
        "clauses": len(parsed_doc.clauses),
    })

    return {
        **state,
        "parsed_doc": parsed_doc,
        "iteration": 0,
        "retrieval_threshold": 0.10,
    }


def retrieve_context_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    iteration = state.get("iteration", 0)
    base_threshold = state.get("retrieval_threshold", 0.10)

    threshold = max(0.05, base_threshold - iteration * 0.05)
    logger.info(f"[retrieve_context] iter={iteration}, threshold={threshold:.3f}")

    retrieval_agent = RAGRetrievalAgent(
        persist_directory="vectorstore",
        threshold=threshold,
        k=5,
    )

    parsed_doc = state["parsed_doc"]
    context_map: dict[str, list[RetrievalResultDTO]] = {}

    for clause in parsed_doc.clauses:
        context_map[clause.id] = retrieval_agent.retrieve(
            clause=clause,
            k=5,
            threshold=threshold,
        )

    empty_count = sum(1 for chunks in context_map.values() if not chunks)
    logger.info(f"[retrieve_context] {empty_count}/{len(context_map)} clauze fara context")

    _log_node("retrieve_context", t0, extra={
        "threshold": threshold,
        "clauses_with_context": len(context_map) - empty_count,
        "clauses_without_context": empty_count,
    })

    return {
        **state,
        "context_map": context_map,
        "retrieval_threshold": threshold,
    }


def quality_check_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    parsed_doc = state["parsed_doc"]
    context_map = state.get("context_map", {})
    iteration = state.get("iteration", 0)

    total = len(parsed_doc.clauses)
    if total == 0:
        should_retry = False
    else:
        empty_count = sum(1 for c in parsed_doc.clauses if not context_map.get(c.id))
        empty_rate = empty_count / total
        should_retry = empty_rate > UNKNOWN_RATE_THRESHOLD and iteration < MAX_ITER
        logger.info(f"[quality_check] empty_rate={empty_rate:.2%}, should_retry={should_retry}")

    _log_node("quality_check", t0, extra={"should_retry": should_retry, "iteration": iteration})

    return {
        **state,
        "should_retry_retrieval": should_retry,
        "iteration": iteration + 1 if should_retry else iteration,
    }


def assess_risk_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    logger.info("[assess_risk] start")

    agent = RiskAssessmentAgent()
    parsed_doc = state["parsed_doc"]
    context_map = state.get("context_map", {})

    risk_map = agent.generate_outputs(
        parsed_doc=parsed_doc,
        context_map=context_map,
        output_prefix="data/report",
    )

    _log_node("assess_risk", t0, extra={"clauses_assessed": len(risk_map)})

    return {**state, "risk_map": risk_map}


def flag_high_risk_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    risk_map = state.get("risk_map", {})

    high_risk_count = sum(
        1 for dto in risk_map.values()
        if dto.risk_level == RiskLevel.RIDICAT
    )
    high_risk_alert = high_risk_count >= HIGH_RISK_THRESHOLD

    logger.info(f"[flag_high_risk] clauze RIDICAT={high_risk_count}, alert={high_risk_alert}")
    _log_node("flag_high_risk", t0, extra={"high_risk_count": high_risk_count})

    return {**state, "high_risk_alert": high_risk_alert}


def generate_recommendations_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    logger.info("[generate_recommendations] start")

    agent = RecommendationAgent()
    parsed_doc = state["parsed_doc"]
    context_map = state.get("context_map", {})
    risk_map = state.get("risk_map", {})

    recommendations: list[RecommendationDTO] = []

    for clause in parsed_doc.clauses:
        risk_dto = risk_map.get(clause.id)
        if not risk_dto:
            continue

        context_chunks = context_map.get(clause.id, [])
        rec = agent.recommend(clause, risk_dto, context_chunks)
        recommendations.append(rec)

    _log_node("generate_recommendations", t0, extra={"recommendations": len(recommendations)})

    return {**state, "recommendations": recommendations}


def compile_report_node(state: WorkflowState) -> WorkflowState:

    t0 = time.time()
    logger.info("[compile_report] start")

    agent = RecommendationAgent()
    report_text = agent.generate_report(
        results=state.get("recommendations", []),
        risk_map=state.get("risk_map", {}),
        output_path="data/raport_final.md",
    )

    _log_node("compile_report", t0)

    return {
        **state,
        "report_path": "data/raport_final.md",
        "report_text": report_text,
    }



def should_retry(state: WorkflowState) -> str:
    if state.get("should_retry_retrieval", False):
        logger.info("[routing] retry retrieval")
        return "retrieve_context"
    logger.info("[routing] continua spre assess_risk")
    return "assess_risk"



def build_workflow() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("parse_document", parse_document_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("assess_risk", assess_risk_node)
    graph.add_node("flag_high_risk", flag_high_risk_node)
    graph.add_node("generate_recommendations", generate_recommendations_node)
    graph.add_node("compile_report", compile_report_node)

    graph.set_entry_point("parse_document")

    graph.add_edge("parse_document", "retrieve_context")
    graph.add_edge("retrieve_context", "quality_check")
    graph.add_edge("assess_risk", "flag_high_risk")
    graph.add_edge("flag_high_risk", "generate_recommendations")
    graph.add_edge("generate_recommendations", "compile_report")
    graph.add_edge("compile_report", END)

    graph.add_conditional_edges(
        "quality_check",
        should_retry,
        {
            "retrieve_context": "retrieve_context",
            "assess_risk": "assess_risk",
        }
    )

    return graph.compile()


def run_workflow(pdf_path: str) -> WorkflowState:

    workflow = build_workflow()

    _export_graph_diagram(workflow)

    initial_state: WorkflowState = {"pdf_path": pdf_path}
    final_state = workflow.invoke(initial_state)

    return final_state


def _export_graph_diagram(workflow) -> None:
    try:
        Path("logs").mkdir(exist_ok=True)
        png = workflow.get_graph().draw_mermaid_png()
        with open("logs/workflow_graph.png", "wb") as f:
            f.write(png)
        logger.info("Diagrama graf exportata: logs/workflow_graph.png")
    except Exception as exc:
        logger.warning(f"Nu am putut exporta diagrama grafului: {exc}")


def _log_node(node_name: str, t0: float, extra: dict = None) -> None:

    try:
        Path("logs").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"logs/run_{timestamp[:8]}.json")

        entry = {
            "node": node_name,
            "duration_sec": round(time.time() - t0, 3),
            "timestamp": datetime.now().isoformat(),
            **(extra or {}),
        }

        if log_file.exists():
            existing = json.loads(log_file.read_text(encoding="utf-8"))
        else:
            existing = []

        existing.append(entry)
        log_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    except Exception:
        pass