import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.dtos import ClauseDTO, ParsedDocumentDTO, RetrievalResultDTO, RiskAssessmentDTO, RiskLevel

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Esti un expert juridic roman specializat in analiza contractelor.
Primesti textul unei clauze contractuale si fragmente din legislatia relevanta (context juridic).
Rolul tau este sa evaluezi riscul clauzei STRICT pe baza contextului furnizat.

REGULI CRITICE:
1. Citeaza NUMAI legi, articole sau documente care apar explicit in contextul de mai jos.
   Nu inventa referinte legislative. Daca nu gasesti o referinta concreta in context, nu o mentiona.
2. Daca contextul juridic este insuficient pentru a evalua clauza, returneaza risk_level: "NECUNOSCUT".
3. Raspunzi EXCLUSIV cu un obiect JSON valid, fara text inainte sau dupa.

Schema JSON obligatorie:
{{
  "risk_level": "RIDICAT" | "MEDIU" | "SCAZUT" | "CONFORM" | "NECUNOSCUT",
  "issues": ["problema 1", "problema 2"],
  "references": ["Sursa exacta din context: articol, document"]
}}

Definitii niveluri de risc:
- RIDICAT: clauza incalca legislatia sau creeaza dezechilibru contractual sever
- MEDIU: clauza are ambiguitati sau omisiuni care pot genera litigii
- SCAZUT: clauza are deficiente minore, dar nu creeaza risc major
- CONFORM: clauza respecta cadrul legislativ si este echilibrata
- NECUNOSCUT: contextul juridic furnizat este insuficient pentru evaluare"""
USER_PROMPT = """Context juridic recuperat din corpus:
{legal_context}

---
Clauza contractuala de evaluat:
Tip: {clause_type}
Sectiune: {clause_section}
Text: {clause_text}

Evalueaza riscul acestei clauze pe baza contextului juridic de mai sus."""


class RiskAssessmentAgent:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm | self.parser

    def assess(
            self,
            clause: ClauseDTO,
            context_chunks: list[RetrievalResultDTO],
    ) -> RiskAssessmentDTO:

        if not context_chunks:
            logger.warning(f"[{clause.id}] context gol — returnez NECUNOSCUT fara apel LLM")
            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=RiskLevel.NECUNOSCUT,
                issues=["Nu a fost gasit context juridic relevant in corpus pentru aceasta clauza."],
                references=[],
                context_was_empty=True,
            )

        legal_context = "\n\n---\n\n".join(
            f"[Sursa: {chunk.source} | Scor similaritate: {chunk.score:.3f}]\n{chunk.text}"
            for chunk in context_chunks
        )

        try:
            raw = self.chain.invoke({
                "legal_context": legal_context,
                "clause_type": clause.type.value,
                "clause_section": clause.section,
                "clause_text": clause.text[:2000],
            })

            risk_level_str = str(raw.get("risk_level", "NECUNOSCUT")).upper()
            try:
                risk_level = RiskLevel(risk_level_str)
            except ValueError:
                logger.warning(f"[{clause.id}] risk_level necunoscut: {risk_level_str!r} — folosesc NECUNOSCUT")
                risk_level = RiskLevel.NECUNOSCUT

            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=risk_level,
                issues=raw.get("issues", []),
                references=raw.get("references", []),
                context_was_empty=False,
            )

        except Exception as exc:
            logger.error(f"[{clause.id}] eroare la parsare LLM: {exc}")
            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=RiskLevel.NECUNOSCUT,
                issues=[f"Eroare la evaluare: {str(exc)[:200]}"],
                references=[],
                context_was_empty=False,
            )

    def generate_outputs(
            self,
            parsed_doc: ParsedDocumentDTO,
            context_map: dict[str, list[RetrievalResultDTO]],
            output_prefix: str = "data/report",
    ) -> dict[str, RiskAssessmentDTO]:

        risk_map: dict[str, RiskAssessmentDTO] = {}

        for clause in parsed_doc.clauses:
            context_chunks = context_map.get(clause.id, [])
            risk_dto = self.assess(clause, context_chunks)
            risk_map[clause.id] = risk_dto
            logger.info(f"[{clause.id}] risk={risk_dto.risk_level.value}")

        Path("data").mkdir(exist_ok=True)
        risks_path = f"{output_prefix}_risks.json"
        with open(risks_path, "w", encoding="utf-8") as f:
            json.dump(
                {cid: dto.model_dump() for cid, dto in risk_map.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        self._save_risk_distribution(risk_map)
        self._analyze_hallucinations(risk_map, context_map)

        return risk_map

    def _save_risk_distribution(self, risk_map: dict[str, RiskAssessmentDTO]) -> None:
        Path("logs").mkdir(exist_ok=True)
        counts = Counter(dto.risk_level.value for dto in risk_map.values())

        order = ["RIDICAT", "MEDIU", "SCAZUT", "CONFORM", "NECUNOSCUT"]
        colors = ["#e03131", "#f08c00", "#f59f00", "#2f9e44", "#868e96"]

        labels = [lvl for lvl in order if lvl in counts]
        values = [counts[lvl] for lvl in labels]
        bar_colors = [colors[order.index(lvl)] for lvl in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set_title("Distributia nivelurilor de risc", fontsize=14, fontweight="bold")
        ax.set_ylabel("Numar clauze")
        ax.set_xlabel("Nivel de risc")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=11,
            )

        fig.tight_layout()
        fig.savefig("logs/risk_distribution.png", dpi=150)
        plt.close(fig)

    def _analyze_hallucinations(
            self,
            risk_map: dict[str, RiskAssessmentDTO],
            context_map: dict[str, list[RetrievalResultDTO]],
    ) -> None:

        Path("logs").mkdir(exist_ok=True)
        lines = ["# Analiza halucinatii — Risk Assessment Agent\n"]
        lines.append("Metodologie: o referinta e considerata posibila halucinatie daca nu apare\n")
        lines.append("ca substring in textul niciunui chunk recuperat din corpus.\n\n")

        hallucination_count = 0
        total_refs = 0

        for clause_id, dto in risk_map.items():
            if not dto.references:
                continue

            chunks = context_map.get(clause_id, [])
            corpus_text = " ".join(chunk.text.lower() for chunk in chunks)

            for ref in dto.references:
                total_refs += 1
                keywords = [w.lower() for w in ref.split() if len(w) > 4]
                found = any(kw in corpus_text for kw in keywords) if keywords else False

                if not found:
                    hallucination_count += 1
                    lines.append(f"[POSIBILA HALUCINATIE] Clauza {clause_id}\n")
                    lines.append(f"  Referinta: {ref}\n")
                    lines.append(f"  Motivatie: niciun cuvant cheie din referinta nu apare in contextul recuperat.\n\n")

        lines.append(f"\n---\nTotal referinte analizate: {total_refs}\n")
        lines.append(f"Posibile halucinatii detectate: {hallucination_count}\n")
        if total_refs > 0:
            lines.append(f"Rata estimata: {hallucination_count/total_refs*100:.1f}%\n")

        with open("logs/hallucinations.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)
