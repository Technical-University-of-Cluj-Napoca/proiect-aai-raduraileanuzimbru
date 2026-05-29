from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ClauseType(str, Enum):
    penalitate = "penalitate"
    obligatie = "obligatie"
    drept = "drept"
    forta_majora = "forta_majora"
    confidentialitate = "confidentialitate"
    reziliere ="reziliere"
    date_personale = "date_personale"
    altele = "altele"

class RiskLevel(str, Enum):
    RIDICAT = "RIDICAT"
    MEDIU = "MEDIU"
    SCAZUT = "SCAZUT"
    CONFORM = "CONFORM"
    NECUNOSCUT = "NECUNOSCUT"

class PartyDTO(BaseModel):
    name: str = ""
    cui_cnp: Optional[str] = None
    address: Optional[str] = None

class SectionDTO(BaseModel):
    title: str
    start_page: int

class ClauseDTO(BaseModel):
    id: str
    section: str
    text: str
    page: int
    type: ClauseType = ClauseType.altele

class DocumentMetadataDTO(BaseModel):
    title: str = ""
    page_count: int = 0
    parties: list[PartyDTO] = Field(default_factory=list)
    signing_date: Optional[str] = None
    effective_date: Optional[str] = None
    value: Optional[str] = None
    duration: Optional[str] = None


class ParsedDocumentDTO(BaseModel):
    metadata: DocumentMetadataDTO
    sections: list[SectionDTO] = Field(default_factory=list)
    clauses: list[ClauseDTO] = Field(default_factory=list)


class RetrievalResultDTO(BaseModel):
    text: str
    source: str
    score: float


class RiskAssessmentDTO(BaseModel):
    clause_id: str
    risk_level: RiskLevel
    issues: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    context_was_empty: bool = False


class RecommendationDTO(BaseModel):
    clause_id: str
    original_text: str
    reformulated_text: str = ""
    explanation: str = ""
    sources: list[str] = Field(default_factory=list)
    candidates: Optional[list[str]] = None