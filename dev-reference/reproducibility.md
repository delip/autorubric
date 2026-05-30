[← Dev Reference Index](index.md)

### Reproducibility & Seed Coordination
`CriterionGrader(seed=...)` controls all non-LLM randomness (option shuffling, few-shot example selection). Auto-generated when `None` so shuffles are always pinned.

- Per-call shuffle RNGs derived from `(master_seed, content_hash, criterion_idx, judge_id)` via SHA-256 — concurrency-safe (no shared mutable state). Helper: `_derive_shuffle_rng()` in `criterion_grader.py`.
- If `FewShotConfig.seed` is unset, it is coordinated from the master seed.
- Few-shot example selection RNGs are derived per `(few_shot_seed, criterion_idx, judge_id)` via the same `_derive_shuffle_rng` helper (a constant `FEW_SHOT_DOMAIN` sits in the item-key slot — few-shot examples are a fixed property of criterion+judge, not item-specific), de-correlating selected examples and their ordering across both judges and criteria. Mirrors option shuffling and supports the ensemble-independence assumption behind the inter-judge agreement metrics; selection stays fully reproducible.
- `CriterionReport.shuffle_order` records the permutation per multi-choice criterion.
- `ExperimentManifest.grader_config` persists `master_seed`, `shuffle_options`, `auto_na_option` for checkpoint reproducibility (via `_serialize_grader_config` in `eval.py`).
- With `auto_na_option=True`, a multi-choice criterion previously lacking an NA option gains one (appended at end), so its shuffle permutation spans one more index and `shuffle_order` grows by one; criteria that already had an NA option are unaffected.
