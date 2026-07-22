# PainConnect-RD

PainConnect-RD is a proposal-stage research scaffold for studying rare pediatric pain disorders with an evidence-grounded mechanistic graph. The initial disease set centers on SCN9A-associated conditions, including primary erythromelalgia, paroxysmal extreme pain disorder, and congenital insensitivity to pain, with planned extension to selected SCN10A-, SCN11A-, PRDM12-, and NTRK1-associated disorders.

This project is intentionally lightweight. It documents the proposed scientific scope, public data sources, schema, evaluation plan, and a minimal software skeleton for literature ingestion and evidence normalization. It should be read as a serious starting point for the grant project, not as a claim that the full rare-disease pipeline is already complete.

## Goal

The core question is whether structured synthesis across rare monogenic pain disorders can reveal shared and opposing mechanisms from variant to ion-channel function to cellular physiology to circuit-level phenotype.

## Initial Scope

- Build a mechanistic evidence schema for rare pediatric pain disorders.
- Ingest public, non-identifiable sources such as PubMed, ClinVar, Orphanet, Mondo, HPO, and Monarch resources.
- Normalize extracted claims into disease, gene, variant, mechanism, anatomy, phenotype, and intervention records.
- Rank candidate cross-disease relationships while preserving source provenance and uncertainty.
- Release the workflow, outputs, and limitations publicly.

## Six-Month Plan

1. Finalize schema, source inventory, and evaluation protocol.
2. Curate and validate SCN9A evidence across the three core disorders.
3. Extend extraction and comparison to adjacent sodium-channel and nociception disorders.
4. Build ranking, contradiction checks, and error analysis.
5. Prepare DisMech-compatible outputs and public benchmark artifacts.
6. Release the first open dataset, graph explorer, and manuscript draft.

## Public Data Sources

- PubMed and PubMed Central
- ClinVar
- Orphanet
- Mondo Disease Ontology
- Human Phenotype Ontology
- Monarch Initiative resources, including DisMech where applicable

## Current Structure

- `docs/schema.md`: proposed graph schema and normalization rules
- `docs/evaluation.md`: success metrics, failure modes, and review plan
- `src/painconnect_rd/`: minimal Python skeleton for source registration and evidence records
- `tests/`: basic checks for schema-level assumptions

## Research Boundary

PainConnect-RD is research software. It is not a diagnostic system, treatment recommender, or clinical decision-support tool. Any mechanistic relationship produced by this workflow must remain a hypothesis until it is supported by cited evidence and external domain review.
