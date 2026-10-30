# Upgrade Notes

## 1.7.0

### Per-criterion prompts put the criterion last

A per-criterion prompt now runs in this order: guidelines, input, reference submission, submission, few-shot examples, criterion. Before 1.7.0 the criterion came first, which kept providers from reusing the submission across an item's calls. The text of each part is unchanged, and only the default `llm_calls="per_criterion"` is affected (meta-rubric evaluation included, as it grades per criterion). `llm_calls="per_item"` prompts and decision-model requests do not change.

MEASURED_NUMBERS

### Your response caches keep working

Cache directories written by earlier releases still load, and their replies are still used, so upgrading does not pay again for calls you have already made. A run reading such a cache can return verdicts generated under the old order. Clear the cache to get replies generated under the new one.

This is for backward compatibility only. We plan to switch entirely to the new layout in 2.0.0, when replies cached under the old layout will no longer be read.

### Resuming a run

Experiment directories from earlier releases still load. Resuming an unfinished one grades its remaining items in the new order, so that run mixes the two layouts. Start a new experiment if you need a single layout.
