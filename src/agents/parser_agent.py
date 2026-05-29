import json
import re
from collections import Counter
from pathlib import Path

from src.dtos import (
    ClauseDTO,
    ClauseType,
    DocumentMetadataDTO,
    ParsedDocumentDTO,
    PartyDTO,
    SectionDTO,
)
from src.tools.pdf_tools import extract_pages_text, get_page_count


class DocumentParserAgent:
    def parse(self, pdf_path: str) -> ParsedDocumentDTO:
        """
        Primeste calea catre un PDF si returneaza documentul structurat.
        Parserul este facut robust: daca nu gaseste un camp, pune None sau lista goala.
        """
        pages = extract_pages_text(pdf_path)
        page_count = get_page_count(pdf_path)
        full_text = "\n".join(page["text"] for page in pages)

        metadata = DocumentMetadataDTO(
            title=self._extract_title(full_text, pdf_path),
            page_count=page_count,
            parties=self._extract_parties(full_text),
            signing_date=self._extract_signing_date(full_text),
            effective_date=self._extract_effective_date(full_text),
            value=self._extract_value(full_text),
            duration=self._extract_duration(full_text),
        )

        sections = self._extract_sections(pages)
        clauses = self._extract_clauses(pages)

        return ParsedDocumentDTO(
            metadata=metadata,
            sections=sections,
            clauses=clauses,
        )

    def save_parsed_json(self, parsed_doc: ParsedDocumentDTO, output_path: str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            parsed_doc.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def save_parser_stats(self, parsed_doc: ParsedDocumentDTO, output_path: str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        type_counts = Counter(clause.type.value for clause in parsed_doc.clauses)

        stats = {
            "title": parsed_doc.metadata.title,
            "page_count": parsed_doc.metadata.page_count,
            "sections_count": len(parsed_doc.sections),
            "clauses_count": len(parsed_doc.clauses),
            "clauses_by_type": dict(type_counts),
        }

        output.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _extract_title(self, full_text: str, pdf_path: str) -> str:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]

        for line in lines[:10]:
            if "contract" in line.lower():
                return line

        return Path(pdf_path).stem

    def _extract_parties(self, full_text: str) -> list[PartyDTO]:
        parties = []

        patterns = [
            r"(?:între|intre)\s+(.+?)\s+(?:și|si)\s+(.+?)(?:,|\n)",
            r"prestator(?:ul)?\s*[:\-]\s*(.+)",
            r"beneficiar(?:ul)?\s*[:\-]\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                groups = match.groups()
                for group in groups:
                    name = self._clean_text(group)
                    if name and len(name) < 200:
                        parties.append(PartyDTO(name=name))

        return parties[:4]

    def _extract_signing_date(self, full_text: str) -> str | None:
        patterns = [
            r"data semnării\s*[:\-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
            r"semnat(?:\s+la)?\s+data\s+de\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
        ]

        return self._first_regex_group(patterns, full_text)

    def _extract_effective_date(self, full_text: str) -> str | None:
        patterns = [
            r"intr[ăa] în vigoare\s+(?:la data de)?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
            r"data intrării în vigoare\s*[:\-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
        ]

        return self._first_regex_group(patterns, full_text)

    def _extract_value(self, full_text: str) -> str | None:
        patterns = [
            r"valoarea contractului\s*(?:este|:)?\s*([0-9\.\s,]+ ?(?:lei|ron|eur|euro))",
            r"pre[tț]ul\s*(?:este|:)?\s*([0-9\.\s,]+ ?(?:lei|ron|eur|euro))",
        ]

        return self._first_regex_group(patterns, full_text)

    def _extract_duration(self, full_text: str) -> str | None:
        patterns = [
            r"durata contractului\s*(?:este|:)?\s*([^.\n]+)",
            r"contractul se încheie pe o perioad[ăa] de\s*([^.\n]+)",
        ]

        return self._first_regex_group(patterns, full_text)

    def _extract_sections(self, pages: list[dict]) -> list[SectionDTO]:
        sections = []

        section_pattern = re.compile(
            r"^\s*((?:Articolul|Art\.|Capitolul|Cap\.|Secțiunea|Sectiunea|Clauza)\s+[0-9IVXLC]+[^\n]*|[0-9]{1,2}\.\s+[A-ZĂÂÎȘȚ][^\n]{3,120})",
            flags=re.IGNORECASE | re.MULTILINE,
        )

        seen_titles = set()

        for page in pages:
            for match in section_pattern.finditer(page["text"]):
                title = self._clean_text(match.group(1))

                if title not in seen_titles:
                    sections.append(
                        SectionDTO(
                            title=title,
                            start_page=page["page"],
                        )
                    )
                    seen_titles.add(title)

        return sections

    def _extract_clauses(self, pages: list[dict]) -> list[ClauseDTO]:
        clauses = []

        clause_pattern = re.compile(
            r"((?:(?:Articolul|Art\.|Clauza)\s+[0-9IVXLC]+|[0-9]{1,2}\.)[^\n]*(?:\n(?!\s*(?:(?:Articolul|Art\.|Clauza)\s+[0-9IVXLC]+|[0-9]{1,2}\.)).*)*)",
            flags=re.IGNORECASE,
        )

        clause_index = 1

        for page in pages:
            text = page["text"]
            matches = list(clause_pattern.finditer(text))

            for match in matches:
                clause_text = self._clean_text(match.group(1))

                if len(clause_text) < 80:
                    continue

                clause_type = self._classify_clause(clause_text)

                clauses.append(
                    ClauseDTO(
                        id=f"clause_{clause_index}",
                        section=self._guess_section(clause_text),
                        text=clause_text,
                        page=page["page"],
                        type=clause_type,
                    )
                )
                clause_index += 1

        if not clauses:
            for page in pages:
                paragraphs = self._fallback_paragraph_split(page["text"])

                for paragraph in paragraphs:
                    if len(paragraph) < 100:
                        continue

                    clause_type = self._classify_clause(paragraph)

                    clauses.append(
                        ClauseDTO(
                            id=f"clause_{clause_index}",
                            section="Necunoscut",
                            text=paragraph,
                            page=page["page"],
                            type=clause_type,
                        )
                    )
                    clause_index += 1

        return clauses

    def _classify_clause(self, text: str) -> ClauseType:
        text_lower = text.lower()

        keywords = {
            ClauseType.penalitate: [
                "penalitate",
                "penalități",
                "intarziere",
                "întârziere",
                "daune-interese",
            ],
            ClauseType.forta_majora: [
                "forță majoră",
                "forta majora",
                "eveniment imprevizibil",
            ],
            ClauseType.confidentialitate: [
                "confidențial",
                "confidential",
                "secret comercial",
            ],
            ClauseType.reziliere: [
                "reziliere",
                "încetare",
                "incetare",
                "denunțare",
                "denuntare",
            ],
            ClauseType.date_personale: [
                "date personale",
                "gdpr",
                "prelucrare",
                "operator",
                "persoană vizată",
            ],
            ClauseType.obligatie: [
                "obligația",
                "obligatia",
                "se obligă",
                "se obliga",
                "trebuie să",
            ],
            ClauseType.drept: [
                "are dreptul",
                "dreptul de a",
                "poate solicita",
            ],
        }

        for clause_type, words in keywords.items():
            if any(word in text_lower for word in words):
                return clause_type

        return ClauseType.altele

    def _guess_section(self, clause_text: str) -> str:
        first_line = clause_text.split("\n")[0].strip()
        return first_line[:120] if first_line else "Necunoscut"

    def _fallback_paragraph_split(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-ZĂÂÎȘȚ])", text)
        return [self._clean_text(p) for p in paragraphs if self._clean_text(p)]

    def _first_regex_group(self, patterns: list[str], text: str) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))

        return None

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()