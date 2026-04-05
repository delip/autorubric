# scripts/

One conversion script per raw dataset plus a merge-and-split script.

- `convert_<dataset>.py` — reads from `../raw/<dataset>/`, writes a `RubricGenerationDataset` JSON to `../converted/`.
- `merge_and_split.py` — combines converted datasets, deduplicates, and produces the final splits in `../splits/`.
