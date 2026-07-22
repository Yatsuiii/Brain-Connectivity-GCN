#!/usr/bin/env python3
"""
Generate synthetic fine-tuning examples for a research-use LLM that summarizes
brain-connectivity GCN outputs.

Output: finetune_data/asd_interpreter.jsonl
Format: HuggingFace chat format (system/user/assistant messages)

The text below consists of author-written templates inspired by broad themes in
ASD neuroscience. It is not patient data, clinician-authored reporting, or
subject-level attribution evidence.
- Reduced DMN coherence (mPFC ↔ PCC)
- Atypical salience network (insula, ACC)
- Reduced social brain connectivity (TPJ, STS, FFA)
- Decreased long-range, increased local connectivity
- Atypical cerebellar-cortical coupling
- Thalamo-cortical dysconnectivity
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

SITES = ["NYU", "USM", "UCLA", "UM", "PITT", "CALTECH", "STANFORD",
         "TRINITY", "YALE", "CMU", "LEUVEN", "KKI", "MAXMUN", "OHSU"]

SYSTEM_PROMPT = (
    "You are a research assistant summarizing outputs from an experimental "
    "functional-connectivity classifier trained on ABIDE I. Clearly distinguish "
    "model outputs from measured subject-level findings. Do not diagnose, recommend "
    "care, assign clinical codes, or claim that generated statements are validated. "
    "State that the output is synthetic and for research demonstration only."
)

# ── ASD network findings (literature-grounded) ─────────────────────────────

ASD_HIGH_CONF = [
    ("Default Mode Network", [
        "Marked reduction in posterior cingulate cortex–medial prefrontal cortex coherence, "
        "a hallmark finding in ASD reflecting disrupted self-referential processing.",
        "Substantially attenuated DMN coherence between the precuneus and ventromedial "
        "prefrontal cortex, consistent with impaired mentalizing capacity.",
        "DMN shows reduced anticorrelation with task-positive networks, suggesting atypical "
        "suppression of self-referential activity during externally-directed cognition.",
    ]),
    ("Salience Network", [
        "Reduced anterior insula–anterior cingulate cortex coupling, impairing "
        "interoceptive awareness and social-emotional salience detection.",
        "Atypical salience network lateralization with decreased right anterior insula "
        "connectivity, associated with reduced social orienting in ASD literature.",
        "Diminished salience network–DMN anticorrelation, consistent with disrupted "
        "switching between internal and external attentional states.",
    ]),
    ("Social Brain", [
        "Hypoconnectivity within the social brain circuit: temporoparietal junction, "
        "superior temporal sulcus, and fusiform face area show reduced mutual coherence.",
        "Reduced superior temporal sulcus connectivity with prefrontal regions, "
        "consistent with atypical biological motion and social cue processing.",
        "Fusiform face area shows decreased functional coupling with amygdala and "
        "orbitofrontal cortex, suggesting disrupted social perception pathways.",
    ]),
    ("Cerebellar–Cortical", [
        "Atypical cerebellar–cerebral cortex functional connectivity, particularly "
        "reduced coupling between cerebellar lobule VII and prefrontal regions.",
        "Decreased cerebellar–default mode connectivity, consistent with reports of "
        "cerebellar contribution to social cognition deficits in ASD.",
        "Bilateral cerebellar hypoconnectivity with frontal and temporal association "
        "cortices, implicating motor-cognitive integration pathways.",
    ]),
    ("Long-Range Connectivity", [
        "Global pattern of decreased long-range connectivity with preserved or "
        "elevated local connectivity — a well-replicated ASD connectivity signature.",
        "Reduced interhemispheric connectivity across frontal, temporal, and parietal "
        "regions, consistent with corpus callosum pathway disruption in ASD.",
        "Decreased frontotemporal connectivity (IFG ↔ STG), relevant to language "
        "and social communication networks implicated in ASD.",
    ]),
]

TC_HIGH_CONF = [
    ("Default Mode Network", [
        "DMN coherence within normal range: intact posterior cingulate–medial "
        "prefrontal coupling with appropriate task-positive anticorrelation.",
        "Preserved DMN integrity with typical precuneus–ventromedial PFC coherence, "
        "consistent with intact self-referential and mentalizing function.",
    ]),
    ("Salience Network", [
        "Anterior insula–ACC coupling within expected range, supporting normal "
        "interoceptive and social-emotional salience processing.",
        "Salience network shows typical bilateral organization and appropriate "
        "anticorrelation with DMN.",
    ]),
    ("Long-Range Connectivity", [
        "Long-range cortico-cortical connectivity within normative range across "
        "frontal, parietal, and temporal association areas.",
        "Frontotemporal and interhemispheric connectivity patterns consistent "
        "with typical neurodevelopmental profile.",
    ]),
    ("Cerebellar–Cortical", [
        "Cerebellar–cortical functional coupling within normal limits, with "
        "intact cerebellar–prefrontal and cerebellar–temporal connectivity.",
        "No significant atypicality in cerebellar connectivity; motor-cognitive "
        "integration pathways appear intact.",
    ]),
]

UNCERTAINTY_NOTES = [
    "The borderline probability warrants cautious interpretation; connectivity "
    "patterns show mixed features not clearly typical of either profile.",
    "Model disagreement across scanner sites suggests this case may lie near "
    "the ASD–TC boundary or represent a subthreshold presentation.",
    "Heterogeneity across site-specific models may reflect genuine "
    "neurodevelopmental variability or a complex clinical picture.",
    "The uncertain prediction may reflect atypical comorbidity patterns "
    "or a clinical presentation not well-represented in the ABIDE I cohort.",
]

CAVEATS = [
    "This is a synthetic research template, not a subject-level measurement or diagnosis.",
    "The classifier is retrospective and has not been prospectively validated for clinical use.",
    "Resting-state fMRI model output alone cannot establish an ASD diagnosis or brain mechanism.",
    "LOSO evaluation reduces reliance on one acquisition site but does not prove site invariance.",
]


def format_per_model(per_model: list[tuple[str, float]]) -> str:
    lines = []
    for site, p in per_model:
        label = "ASD" if p > 0.5 else "TC"
        lines.append(f"  {site}: {label} (p={p:.3f})")
    return "\n".join(lines)


def make_per_model(p_asd: float) -> list[tuple[str, float]]:
    sites = random.sample(["NYU", "USM", "UCLA", "UM"], 4)
    scores = []
    for site in sites:
        noise = random.gauss(0, 0.12)
        p = max(0.01, min(0.99, p_asd + noise))
        scores.append((site, round(p, 3)))
    return scores


def asd_confidence_label(p: float) -> str:
    if p >= 0.75:
        return "HIGH"
    elif p >= 0.6:
        return "MODERATE"
    elif p >= 0.4:
        return "LOW / UNCERTAIN"
    elif p >= 0.25:
        return "MODERATE (TC)"
    else:
        return "HIGH (TC)"


def build_user_input(p_asd: float, per_model: list, site: str, n_tp: int) -> str:
    conf = asd_confidence_label(p_asd)
    agreement = sum(1 for _, p in per_model if p > 0.5)
    return f"""Brain Connectivity GCN Analysis Report
========================================
Acquisition Site : {site}
Timepoints       : {n_tp} TRs
p(ASD)           : {p_asd:.3f}
Confidence Level : {conf}
Model Consensus  : {agreement}/4 site models predict ASD

Per-Model Breakdown (LOSO ensemble):
{format_per_model(per_model)}

Please provide a cautious research-use summary of these model outputs."""


def build_asd_report(p_asd: float, per_model: list, site: str) -> str:
    agreement = sum(1 for _, p in per_model if p > 0.5)
    severity = "pronounced" if p_asd > 0.8 else "moderate" if p_asd > 0.65 else "mild-to-moderate"

    # Pick 2-3 network findings
    n_findings = random.randint(2, 3)
    networks = random.sample(ASD_HIGH_CONF, n_findings)
    findings = []
    for net_name, options in networks:
        findings.append(f"**{net_name}**: {random.choice(options)}")

    consensus_note = (
        f"All four site-specific models concur (consensus {agreement}/4)"
        if agreement == 4
        else f"Strong multi-site consensus ({agreement}/4 models agree)"
        if agreement >= 3
        else f"Partial model agreement ({agreement}/4 models)"
    )

    return f"""## Synthetic Research Summary

**Overall Impression**: Connectivity profile consistent with Autism Spectrum Disorder (p_ASD = {p_asd:.3f}). {severity.capitalize()} atypicality across key functional networks. {consensus_note}, supporting cross-scanner robustness of this finding.

**Network-Level Findings**:
{chr(10).join(f'{i+1}. {f}' for i, f in enumerate(findings))}

**Cross-Site Consistency**: The LOSO ensemble shows {agreement}/4 model scores above 0.5. Residual site, motion, demographic, and preprocessing confounding may remain.

**Research Context**: This synthetic narrative is based on broad literature themes and the model score; it is not derived from validated subject-level network attribution.

**Limitation**: {random.choice(CAVEATS)}"""


def build_tc_report(p_asd: float, per_model: list, site: str) -> str:
    agreement = sum(1 for _, p in per_model if p <= 0.5)
    n_findings = random.randint(2, 3)
    networks = random.sample(TC_HIGH_CONF, min(n_findings, len(TC_HIGH_CONF)))
    findings = []
    for net_name, options in networks:
        findings.append(f"**{net_name}**: {random.choice(options)}")

    return f"""## Synthetic Research Summary

**Overall Impression**: Connectivity profile within typical range (p_ASD = {p_asd:.3f}). No significant functional connectivity atypicalities detected across primary networks associated with ASD. {agreement}/4 site-specific models concur with a typical neurodevelopmental classification.

**Network-Level Findings**:
{chr(10).join(f'{i+1}. {f}' for i, f in enumerate(findings))}

**Cross-Site Consistency**: {agreement}/4 LOSO model scores are at or below 0.5. This does not establish the absence of ASD or acquisition-site effects.

**Research Context**: This is a classifier output relative to its training distribution, not a clinical normative assessment or validated biomarker result.

**Limitation**: {random.choice(CAVEATS)}"""


def build_uncertain_report(p_asd: float, per_model: list, site: str) -> str:
    agreement = sum(1 for _, p in per_model if p > 0.5)

    asd_net = random.choice(ASD_HIGH_CONF)
    tc_net  = random.choice(TC_HIGH_CONF)

    return f"""## Synthetic Research Summary

**Overall Impression**: Indeterminate model output (p_ASD = {p_asd:.3f}). The ensemble shows mixed scores ({agreement}/4 above 0.5), so the research system should abstain from a binary summary.

**Mixed Network Findings**:
1. **{asd_net[0]} (Atypical)**: {random.choice(asd_net[1])}
2. **{tc_net[0]} (Within Range)**: {random.choice(tc_net[1])}

**Model Disagreement**: Cross-site model disagreement ({agreement}/4 above 0.5) may reflect a score near the model boundary, dataset shift, or residual confounding.

**Note**: {random.choice(UNCERTAINTY_NOTES)}

**Limitation**: {random.choice(CAVEATS)}"""


def generate_example(p_asd: float | None = None) -> dict:
    if p_asd is None:
        # Sample p_asd with realistic distribution — more extremes than middle
        bucket = random.random()
        if bucket < 0.35:
            p_asd = random.uniform(0.65, 0.97)   # clear ASD
        elif bucket < 0.60:
            p_asd = random.uniform(0.03, 0.35)   # clear TC
        else:
            p_asd = random.uniform(0.35, 0.65)   # uncertain

    p_asd = round(p_asd, 3)
    per_model = make_per_model(p_asd)
    site = random.choice(SITES)
    n_tp = random.randint(110, 316)

    user_msg  = build_user_input(p_asd, per_model, site, n_tp)

    if p_asd >= 0.6:
        assistant_msg = build_asd_report(p_asd, per_model, site)
    elif p_asd <= 0.4:
        assistant_msg = build_tc_report(p_asd, per_model, site)
    else:
        assistant_msg = build_uncertain_report(p_asd, per_model, site)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main() -> None:
    out_dir = Path("finetune_data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "asd_interpreter.jsonl"

    n = 600
    examples = [generate_example() for _ in range(n)]

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Quick stats
    p_vals = [
        float(ex["messages"][1]["content"].split("p(ASD)")[1].split(":")[1].split("\n")[0].strip())
        for ex in examples
    ]
    asd_count = sum(1 for p in p_vals if p >= 0.6)
    tc_count  = sum(1 for p in p_vals if p <= 0.4)
    unc_count = n - asd_count - tc_count

    print(f"Generated {n} examples → {out_path}")
    print(f"  ASD (p≥0.6) : {asd_count}")
    print(f"  TC  (p≤0.4) : {tc_count}")
    print(f"  Uncertain   : {unc_count}")
    print(f"\nSample user input:\n{'-'*50}")
    ex = examples[0]
    print(ex["messages"][1]["content"])
    print(f"\nSample assistant output:\n{'-'*50}")
    print(ex["messages"][2]["content"])


if __name__ == "__main__":
    main()
