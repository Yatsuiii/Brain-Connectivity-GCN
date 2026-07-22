# Contributing

Thank you for improving BrainConnect-ASD. Contributions that strengthen
reproducibility, evaluation, documentation, accessibility, and research safety
are especially welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

## Pull requests

Keep each pull request focused and describe:

- the problem and proposed change;
- the data, split, seed, and preprocessing used for experimental results;
- tests added or updated;
- expected compatibility or migration effects; and
- limitations or failed cases discovered during the work.

Do not commit ABIDE data, identifiable information, credentials, generated
patient-like records, or artifacts whose licenses do not permit redistribution.

## Reporting results

Do not select or tune models using the test partition. Report the evaluation
scope next to every metric. For the existing project, distinguish the four-site
LOSO result (AUC 0.7872, N=529) from the broader 20-site LOSO experiment
(AUC 0.7298, N=1,102).

Claims of clinical validity require appropriate prospective evidence and are
out of scope for this repository's current model.
