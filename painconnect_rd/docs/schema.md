# Mechanistic Schema

PainConnect-RD represents each disease mechanism as a traceable chain of evidence rather than a free-text summary. Every claim should remain attached to a source, identifiers, and an explicit confidence level.

## Core Entities

### Disease

- Canonical label
- Synonyms
- Ontology identifiers: Mondo, Orphanet, OMIM when available
- Disease class

### Gene

- HGNC symbol
- Stable identifier
- Functional role summary

### Variant

- HGVS or ClinVar-style representation when available
- Variant effect direction: gain of function, loss of function, uncertain, mixed
- Inheritance pattern when reported

### Mechanism

- Protein or channel effect
- Cellular consequence
- Tissue or anatomical context
- Circuit-level interpretation when evidence exists

### Phenotype

- HPO identifier
- Clinical pain phenotype
- Age of onset when reported
- Supporting context

### Intervention

- Intervention label
- Mechanistic rationale
- Reported response or outcome

### Evidence Record

- Source identifier
- Source type
- Extracted claim text
- Structured claim fields
- Confidence
- Notes on ambiguity or contradiction

## Normalized Claim Chain

The default target structure is:

`disease -> gene -> variant -> molecular effect -> cellular effect -> anatomical or circuit context -> phenotype -> intervention evidence`

Not every record will populate every field. Missing evidence must remain missing rather than inferred.

## Confidence Levels

- `high`: direct, well-supported relationship with clear source grounding
- `moderate`: relationship supported but incomplete or partially indirect
- `low`: suggestive evidence requiring verification
- `conflicted`: explicit disagreement across sources

## Validation Rules

- Every public claim must cite at least one source identifier.
- Ontology identifiers must pass format checks.
- Variant direction must not be inferred if the paper is ambiguous.
- Disease synonyms must map back to one canonical record.
- Contradictory records should be retained and labeled, not silently merged.
