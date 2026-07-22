from dataclasses import dataclass, field


@dataclass(frozen=True)
class PublicSource:
    name: str
    category: str
    access: str
    notes: str = ""


@dataclass(frozen=True)
class EvidenceRecord:
    disease: str
    gene: str
    claim_type: str
    source_id: str
    source_name: str
    confidence: str
    notes: str = ""


@dataclass(frozen=True)
class RareDiseaseProfile:
    disease: str
    genes: tuple[str, ...]
    phenotypes: tuple[str, ...]
    evidence: tuple[EvidenceRecord, ...] = field(default_factory=tuple)
