from .models import EvidenceRecord, PublicSource, RareDiseaseProfile


def public_source_catalog() -> tuple[PublicSource, ...]:
    """Registry of the public sources planned for the initial project scope."""
    return (
        PublicSource(
            name="PubMed",
            category="literature",
            access="public",
            notes="Primary entry point for biomedical abstracts and metadata.",
        ),
        PublicSource(
            name="PubMed Central",
            category="literature",
            access="public",
            notes="Open full-text subset for deeper evidence extraction.",
        ),
        PublicSource(
            name="ClinVar",
            category="variant",
            access="public",
            notes="Variant assertions and supporting clinical context.",
        ),
        PublicSource(
            name="Orphanet",
            category="disease ontology",
            access="public",
            notes="Rare disease definitions and cross-references.",
        ),
        PublicSource(
            name="Mondo",
            category="disease ontology",
            access="public",
            notes="Canonical disease alignment across vocabularies.",
        ),
        PublicSource(
            name="HPO",
            category="phenotype ontology",
            access="public",
            notes="Phenotype standardization for clinical findings.",
        ),
        PublicSource(
            name="Monarch / DisMech",
            category="knowledge graph",
            access="public",
            notes="Target integration surface for mechanistic outputs.",
        ),
    )


def build_stub_profiles() -> tuple[RareDiseaseProfile, ...]:
    """Seed disease profiles for the initial SCN9A-centered scope."""
    return (
        RareDiseaseProfile(
            disease="Primary erythromelalgia",
            genes=("SCN9A",),
            phenotypes=("burning pain", "erythema", "heat-triggered attacks"),
            evidence=(
                EvidenceRecord(
                    disease="Primary erythromelalgia",
                    gene="SCN9A",
                    claim_type="gain_of_function",
                    source_id="seed:scn9a-pepd-001",
                    source_name="manual-seed",
                    confidence="low",
                    notes="Placeholder record for pipeline wiring only.",
                ),
            ),
        ),
        RareDiseaseProfile(
            disease="Paroxysmal extreme pain disorder",
            genes=("SCN9A",),
            phenotypes=("rectal pain", "ocular pain", "autonomic attacks"),
        ),
        RareDiseaseProfile(
            disease="Congenital insensitivity to pain",
            genes=("SCN9A", "PRDM12", "NTRK1"),
            phenotypes=("absent pain perception", "self-injury", "late diagnosis"),
        ),
    )
