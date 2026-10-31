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

### Few-shot examples are cached

We measured few-shot grading (five examples each) on 40 RiceChem answers and 30 ELLIPSE essays, on 500 criterion calls per judge. Cached prompt tokens and cost, per-criterion, compared with the layout without this change:

| Judge | Cached before | Cached after | Cost before | Cost after |
|---|---|---|---|---|
| gpt-5.6-luna | 53% | 89% | $0.354 | $0.207 |
| claude-haiku-5-5 | 56% | 88% | $0.244 | $0.155 |
| gemini-3.8-flash | 2% | 46% | $3.571 | $2.860 |
| deepseek-v4.1-flash | 42% | 81% | $0.616 | $0.724 |

DeepSeek's cost rose because it wrote more output in that run, not because caching got worse. Gemini caches only where the prefix reaches 4,096 tokens: 88% on ELLIPSE, 4% on RiceChem's shorter answers. `per_item` costs fell too, from $0.152 to $0.095 on Luna, $0.091 to $0.064 on Haiku, $1.782 to $1.530 on Gemini and $0.256 to $0.232 on DeepSeek.

Accuracy held in these runs: on RiceChem it moved by at most 1.5 points in either direction, and on ELLIPSE it moved by −3.8 to +8.4 points across judges. Some verdicts change. Luna's matched between layouts on 87% of RiceChem and 80% of ELLIPSE judgments, and Haiku's on only 42% of ELLIPSE, whose six-level scales make exact agreement hard. We ran no repeat here, so we cannot say how much of that is run-to-run noise.

### Meta-rubric evaluation is told what it is reading

The meta-judge now receives an instruction that the submission is a rubric, not a response to grade. Without it, judges could fault a rubric for containing no response. Meta-rubric scores can change, and replies cached for those requests are not reused.

### Running out of credits raises

A 402 Payment Required from an LLM provider or from TypeSafe, which says the account has no credits left, cannot cover the request (OpenRouter also answers 402 when the remaining balance is too low for one long request), or has a billing problem, is now a judge configuration error, like a rejected key. The grade raises a `ValueError` that names the judge and says the provider answered 402, the call is not retried, and `fill_ground_truth` raises it too. Before, each such call abstained as an `infrastructure` failure, so a run that ran out of credits went on to the end, and under the default `SKIP` strategy every item it graded after that point had no score (under `ZERO`, `PARTIAL` or `FAIL` those items were scored from the abstentions). See [Failures](cookbook/llm-judges.md#standalone-use).

### Your response caches keep working

Cache directories written by earlier releases still load, and their replies are still used, so upgrading does not pay again for calls you have already made. A run reading such a cache can return verdicts generated under the old order. Clear the cache to get replies generated under the new one.

This is for backward compatibility only. We plan to switch entirely to the new layout in 2.0.0, when replies cached under the old layout will no longer be read.

### Resuming a run

Experiment directories from earlier releases still load. Resuming an unfinished one grades its remaining items in the new order, so that run mixes the two layouts. Start a new experiment if you need a single layout.
