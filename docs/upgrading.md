# Upgrade Notes

## 1.7.0

### Per-criterion prompts put the criterion last

A per-criterion prompt now runs in this order: guidelines, input, reference submission, submission, criterion. With few-shot examples, which differ per criterion, it runs guidelines, examples, criterion, input, reference submission, submission, and a multi-choice prompt puts its options after the question. Few-shot examples also name a multi-choice option by label rather than by number, in `per_item` prompts too, where the examples now precede the criteria. Before 1.7.0 the criterion came first, which kept providers from reusing the submission across an item's calls. The text of each part is unchanged, and without examples only the default `llm_calls="per_criterion"` is affected (meta-rubric evaluation included, as it grades per criterion). Decision-model requests do not change.

We re-ran nine datasets (5,003 criterion judgments) with each judge and compared the result with one call per item. Cost fell for the two judges whose caches these prompts could reach. Gemini's prompts here were below its 4,096-token minimum for caching, so its cost did not change, though longer submissions and images will benefit.

| Judge | Cost before | Cost after | Before, relative to `per_item` | After, relative to `per_item` | Cached prompt tokens before | Cached prompt tokens after |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | $1.84 | $1.40 | 1.82x | 1.39x | 80% | 91% |
| deepseek-v4.1-flash | $4.18 | $3.23 | 1.23x | 0.95x | 68% | 82% |
| gemini-3.8-flash | $20.59 | $20.68 | 1.79x | 1.79x | none | about 1% |

Expect some verdicts to change. On a 10% repeat, two runs of the new layout agreed on 86.4% of Luna's verdicts, 86.1% of DeepSeek's and 96.4% of Gemini's, while the old and new layouts agreed on 81.1%, 77.9% and 93.6%. So the order itself moves verdicts, beyond run-to-run noise. Accuracy against the labels changed by −0.4 points for Luna (95% interval −1.3 to +0.5), +0.5 for DeepSeek (−0.5 to +1.5) and −1.4 for Gemini (−2.1 to −0.8). If you compare scores across this upgrade, re-grade a sample with both versions first.

### Meta-rubric evaluation is told what it is reading

The meta-judge now receives an instruction that the submission is a rubric, not a response to grade. Without it, judges could fault a rubric for containing no response. Meta-rubric scores can change, and replies cached for those requests are not reused.

### Your response caches keep working

Cache directories written by earlier releases still load, and their replies are still used, so upgrading does not pay again for calls you have already made. A run reading such a cache can return verdicts generated under the old order. Clear the cache to get replies generated under the new one.

This is for backward compatibility only. We plan to switch entirely to the new layout in 2.0.0, when replies cached under the old layout will no longer be read.

### Resuming a run

Experiment directories from earlier releases still load. Resuming an unfinished one grades its remaining items in the new order, so that run mixes the two layouts. Start a new experiment if you need a single layout.
