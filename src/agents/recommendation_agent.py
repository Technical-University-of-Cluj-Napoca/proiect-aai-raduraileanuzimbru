import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.dtos import (
    ClauseDTO,
    RecommendationDTO,
    RetrievalResultDTO,
    RiskAssessmentDTO,
    RiskLevel,
)

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_REFORMULARE = """Esti un jurist roman expert in redactarea contractelor.
Primesti o clauza contractuala cu probleme si contextul juridic relevant.
Reformuleaza clauza pentru a elimina riscurile identificate, respectand:
- Stilul juridic formal specific dreptului roman
- Echilibrul contractual intre parti
- Referintele legislative din contextul furnizat (nu inventa alte legi)
- Claritatea si precizia termenilor juridici

Returneaza EXCLUSIV textul reformulat al clauzei, fara explicatii suplimentare."""

USER_REFORMULARE = """Context juridic:
{legal_context}

Clauza originala (tip: {clause_type}):
{clause_text}

Probleme identificate:
{issues}

Reformuleaza clauza pentru a elimina problemele mentionate."""

SYSTEM_ALEGERE = """Esti un jurist roman senior care evalueaza reformulari alternative ale unei clauze contractuale.
Alege varianta cea mai buna din perspectiva juridica si explica alegerea.
Returneaza raspunsul in formatul:
VARIANTA ALEASA: [numarul variantei, 1-3]
EXPLICATIE: [motivarea alegerii in 2-3 fraze]"""

USER_ALEGERE = """Clauza originala:
{clause_text}

Probleme de rezolvat:
{issues}

Variante de reformulare:
{candidates}

Care este cea mai buna varianta juridica si de ce?"""


class RecommendationAgent:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.llm_chooser = ChatOpenAI(model=model, temperature=0)

        self.reform_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_REFORMULARE),
            ("human", USER_REFORMULARE),
        ])
        self.choose_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_ALEGERE),
            ("human", USER_ALEGERE),
        ])

    def recommend(
            self,
            clause: ClauseDTO,
            risk_assessment: RiskAssessmentDTO,
            context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:

        if risk_assessment.risk_level not in (RiskLevel.RIDICAT, RiskLevel.MEDIU):
            return RecommendationDTO(
                clause_id=clause.id,
                original_text=clause.text,
                reformulated_text="",
                explanation="Clauza nu necesita reformulare (risc scazut sau conform).",
                sources=[],
            )

        legal_context = self._build_legal_context(context_chunks)
        issues_str = "\n".join(f"- {issue}" for issue in risk_assessment.issues) or "- Risc juridic general"

        input_vars = {
            "legal_context": legal_context,
            "clause_type": clause.type.value,
            "clause_text": clause.text[:1500],
            "issues": issues_str,
        }

        if risk_assessment.risk_level == RiskLevel.RIDICAT:
            return self._recommend_with_self_consistency(clause, risk_assessment, input_vars, context_chunks)
        else:
            return self._recommend_single(clause, risk_assessment, input_vars, context_chunks)

    def _recommend_single(
            self,
            clause: ClauseDTO,
            risk_assessment: RiskAssessmentDTO,
            input_vars: dict,
            context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:
        try:
            response = (self.reform_prompt | self.llm).invoke(input_vars)
            reformulated = response.content.strip()
        except Exception as exc:
            logger.error(f"[{clause.id}] eroare reformulare: {exc}")
            reformulated = ""

        return RecommendationDTO(
            clause_id=clause.id,
            original_text=clause.text,
            reformulated_text=reformulated,
            explanation=f"Reformulare pentru risc {risk_assessment.risk_level.value}: {', '.join(risk_assessment.issues[:2])}",
            sources=[chunk.source for chunk in context_chunks[:3]],
        )

    def _recommend_with_self_consistency(
            self,
            clause: ClauseDTO,
            risk_assessment: RiskAssessmentDTO,
            input_vars: dict,
            context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:
        candidates = []

        for i in range(3):
            try:
                response = (self.reform_prompt | self.llm).invoke(input_vars)
                candidates.append(response.content.strip())
                logger.info(f"[{clause.id}] varianta {i+1} generata ({len(candidates[-1])} chars)")
            except Exception as exc:
                logger.error(f"[{clause.id}] eroare varianta {i+1}: {exc}")

        if not candidates:
            return RecommendationDTO(
                clause_id=clause.id,
                original_text=clause.text,
                reformulated_text="",
                explanation="Eroare la generarea reformularilor.",
                sources=[],
            )

        if len(candidates) == 1:
            chosen = candidates[0]
            explanation = "O singura varianta generata (celelalte au esuat)."
        else:
            candidates_formatted = "\n\n".join(
                f"Varianta {i+1}:\n{c}" for i, c in enumerate(candidates)
            )
            try:
                choice_response = (self.choose_prompt | self.llm_chooser).invoke({
                    "clause_text": clause.text[:800],
                    "issues": "\n".join(f"- {iss}" for iss in risk_assessment.issues[:3]),
                    "candidates": candidates_formatted,
                })
                choice_text = choice_response.content.strip()

                chosen_idx = 0
                explanation = choice_text

                for i in range(len(candidates), 0, -1):
                    if str(i) in choice_text:
                        chosen_idx = i - 1
                        break

                if "EXPLICATIE:" in choice_text:
                    explanation = choice_text.split("EXPLICATIE:")[-1].strip()

                chosen = candidates[chosen_idx]
                logger.info(f"[{clause.id}] self-consistency: varianta {chosen_idx+1} aleasa")

            except Exception as exc:
                logger.error(f"[{clause.id}] eroare alegere self-consistency: {exc}")
                chosen = candidates[0]
                explanation = "Prima varianta aleasa (eroare la pasul de selectie)."

        return RecommendationDTO(
            clause_id=clause.id,
            original_text=clause.text,
            reformulated_text=chosen,
            explanation=explanation,
            sources=[chunk.source for chunk in context_chunks[:3]],
            candidates=candidates,
        )

    def generate_report(
            self,
            results: list[RecommendationDTO],
            risk_map: dict[str, RiskAssessmentDTO],
            output_path: str = "data/raport_final.md",
    ) -> str:

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Raport de Analiza Juridica a Contractului\n\n"]
        lines.append("---\n\n")

        risk_counts = {}
        for dto in risk_map.values():
            risk_counts[dto.risk_level.value] = risk_counts.get(dto.risk_level.value, 0) + 1

        lines.append("## Sumar executiv\n\n")
        lines.append("| Nivel de risc | Nr. clauze |\n")
        lines.append("|--------------|------------|\n")
        for level in ["RIDICAT", "MEDIU", "SCAZUT", "CONFORM", "NECUNOSCUT"]:
            if level in risk_counts:
                lines.append(f"| {level} | {risk_counts[level]} |\n")
        lines.append("\n")

        if risk_counts.get("RIDICAT", 0) > 0:
            lines.append(f"> ⚠️ **Atentie:** {risk_counts['RIDICAT']} clauze cu risc RIDICAT necesita revizuire urgenta.\n\n")

        lines.append("---\n\n")
        lines.append("## Analiza detaliata pe clauze\n\n")

        level_order = {"RIDICAT": 0, "MEDIU": 1, "SCAZUT": 2, "CONFORM": 3, "NECUNOSCUT": 4}
        sorted_results = sorted(
            results,
            key=lambda r: level_order.get(
                risk_map.get(r.clause_id, RiskAssessmentDTO(
                    clause_id=r.clause_id, risk_level=RiskLevel.NECUNOSCUT
                )).risk_level.value,
                5
            )
        )

        for rec in sorted_results:
            risk_dto = risk_map.get(rec.clause_id)
            if not risk_dto:
                continue

            level = risk_dto.risk_level.value
            emoji = {"RIDICAT": "🔴", "MEDIU": "🟡", "SCAZUT": "🟢", "CONFORM": "✅", "NECUNOSCUT": "⚪"}.get(level, "⚪")

            lines.append(f"### {emoji} {rec.clause_id} — {level}\n\n")

            lines.append("**Text original:**\n\n")
            lines.append(f"> {rec.original_text[:500]}{'...' if len(rec.original_text) > 500 else ''}\n\n")

            if risk_dto.issues:
                lines.append("**Probleme identificate:**\n\n")
                for issue in risk_dto.issues:
                    lines.append(f"- {issue}\n")
                lines.append("\n")

            if risk_dto.references:
                lines.append("**Referinte legislative:**\n\n")
                for ref in risk_dto.references:
                    lines.append(f"- {ref}\n")
                lines.append("\n")

            if rec.reformulated_text:
                lines.append("**Reformulare propusa:**\n\n")
                lines.append(f"> {rec.reformulated_text}\n\n")
                lines.append(f"*{rec.explanation}*\n\n")

                if rec.sources:
                    lines.append("**Surse corpus:**\n\n")
                    for src in rec.sources:
                        lines.append(f"- `{src}`\n")
                    lines.append("\n")

            lines.append("---\n\n")

        report_text = "".join(lines)
        Path(output_path).write_text(report_text, encoding="utf-8")
        logger.info(f"Raport generat: {output_path}")
        return report_text

    def _build_legal_context(self, context_chunks: list[RetrievalResultDTO]) -> str:
        if not context_chunks:
            return "Nu a fost gasit context juridic relevant."
        return "\n\n---\n\n".join(
            f"[Sursa: {chunk.source}]\n{chunk.text}" for chunk in context_chunks
        )
