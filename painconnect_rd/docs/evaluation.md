# Evaluation Plan

The project will be judged on scientific usefulness, evidence traceability, and failure transparency rather than on raw extraction volume alone.

## Primary Success Metrics

- Evidence-backed mechanistic graphs for 6 to 10 rare pain disorders
- At least 500 processed source records in the initial release
- At least 90% valid citation attachment in manual spot checks
- At least 80% claim-support accuracy in blinded review of sampled records
- At least 60% reduction in literature review time relative to fully manual curation

## Review Questions

- Does each structured claim remain grounded in a real supporting source?
- Are gain-of-function and loss-of-function distinctions preserved correctly?
- Are disease labels, variants, and phenotypes normalized consistently?
- Are contradictions surfaced explicitly instead of being averaged away?
- Do cross-disease links produce biologically interpretable hypotheses?

## Failure Modes To Track

- Unsupported causal language
- Variant-direction errors
- Disease-name confusion
- False equivalence across related pain syndromes
- Missing negative evidence
- Hallucinated ontology identifiers
- Overstated circuit claims when only peripheral evidence exists

## Release Standards

- Public outputs must include source provenance.
- Error analysis must be published with the first release.
- Limitations must be stated alongside any benchmark numbers.
- No disease relationship should be presented as validated without external review.
