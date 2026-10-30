# Per-criterion prompt caching

## Why

Per-criterion grading (`llm_calls="per_criterion"`, the default) stays the recommended mode.
It costs about 1.8x holistic grading (`llm_calls="per_item"`) on Luna and Gemini and 1.23x on
DeepSeek over nine datasets (`jev_experiments/results_per_item/paper/mode_comparison.json`).
Today's user prompt puts the criterion before the submission, so an item's calls share only
the system prompt and guidelines as a cacheable prefix. Image inputs arrive right after this
work and make input the dominant cost, so the submission has to be in the shared prefix.

## Goals

- The default per-criterion prompt puts everything shared by an item's calls first and the
  criterion last.
- OpenAI-style automatic caching and Anthropic breakpoints can both reuse that shared prefix.
- An item's calls reach a warm cache, instead of all missing together in one parallel burst.
- Measured cost, cached prompt tokens, wall time and verdict agreement, per mode, from a
  consented run.
- The branch merges to `main` first in 1.7.0. `autorubric-image` is then synced onto `main`.

## Non-goals

- Prompts under `llm_calls="per_item"`, decision-model requests and revision prompts. All stay
  byte-identical. Meta-rubric evaluation grades through `CriterionGrader` per criterion, so its
  prompts take the new layout like any other per-criterion call.
- A different few-shot selection. A single example set shared across criteria would cache
  better but changes grading behaviour; it is a possible later change.
- Gemini explicit context caching. Live probes of Gemini 3.8 Flash found its automatic cache
  was not readable until about a minute after a call, and only for a repeated prefix, even
  well above its 4,096-token minimum. That is too late for an item's other calls, so Gemini
  gets one user message and no warm-first. Explicit caching is a possible later change for
  image submissions. The docs state the limit.
- Changing the `autorubric-image` branch.

## Anti-goals

- An old response cache directory or experiment directory that fails to open or load.
- A reply served from the cache for different content. A pre-1.7.0 reply to the same content
  in the old order is the one permitted case (Decision 4).
- Any new paid call without the author's per-run consent and an estimate.

## Decisions made with the author

1. **New default.** There is no opt-in. Input cost will dominate with images, so the layout
   does not wait on the measurement. The measurement still reports verdict agreement, and the
   version note carries it.
2. **Criterion last.** It is the one part guaranteed to change from call to call.
3. **Few-shot examples sit just before the criterion**, after the submission. Examples are
   fixed across items but are drawn per criterion and show that criterion's verdict, so before
   the submission they would end the shared prefix. Selection and content do not change.
4. **Previously cached directories still load, and their replies are still reused.** Formats
   do not change. A read-only fallback finds pre-1.7.0 replies for the same content, so an
   upgrade does not pay for those calls again. It is always on, with no setting.

## Design

### Prompt order

The per-criterion user content becomes:

    shared:   guidelines, <input>, <reference_submission>, <submission>
    tail:     <examples> (when there are any), <criterion_type>/<criterion> or the multi-choice
              criterion block with its options

The tags and the text inside each block are unchanged. Option shuffling stays inside the tail,
with the same `_presented_criterion` permutation and RNG key: the content hash that keys it is
of the submission and does not depend on position. The four builders
(`build_user_prompt`, `build_few_shot_user_prompt`, `build_multi_choice_user_prompt`,
`build_multi_choice_few_shot_user_prompt`) change order together, and goldens under
`tests/golden/` change deliberately in one commit, reviewed as a diff.

### Sending two blocks

A single user message does not work on GPT-5.6 and later: automatic caching looks up prefixes
at message endings (the latest eligible message and up to 20 earlier ones), so a prefix ending
inside a message cannot be reused. The shared block and the tail are therefore two consecutive
user messages for models known to cache prompts. Other models keep one user message, since
splitting gains them nothing and some OpenAI-compatible servers refuse consecutive user
messages. On Anthropic models the shared message's content carries
`cache_control: {"type": "ephemeral"}`, alongside the existing marker on the system message
(two of the four allowed breakpoints). LiteLLM merges consecutive user messages for providers
that need alternating roles; the implementation confirms this for Anthropic and Gemini, and
that `cache_control` survives the merge.

`LLMClient.generate` keeps accepting one `user_prompt` and gains an optional keyword for the
shared block, so other callers (per-item calls, meta-rubric, revision) send exactly what they
send today. `is_anthropic` misses routed ids, so it widens to `openrouter/anthropic/...` by a
provider check. `supports_prompt_caching` cannot be that check: it is true for OpenAI and
Gemini models too, and on Gemini LiteLLM acts on `cache_control` by creating explicit caches.
Bedrock and Vertex Claude ids stay sent as they are today (single user message content
unmarked), because the current docstring promises it and no measurement covers them; they
still get the new order.

OpenAI models from GPT-5.6 on need one more step. Live probes showed automatic mode writes the
whole prompt at the end of the last message on every call, at 1.25x the input rate, and calls
differing in that message never read it. For split requests only, these models get
`prompt_cache_breakpoint: {"mode": "explicit"}` on the system message and the shared block and
`prompt_cache_options: {"mode": "explicit"}` in `extra_body`, on Chat Completions. Later calls
then read the whole shared prefix. Requests that are not split stay in automatic mode.

### Response cache key

`_cache_key` hashes the shared block and the tail joined in prompt order, in the existing key
format. Two requests differing only in where the split falls are different requests only if
their text differs, so the key needs no field of its own. No key changes for requests that do
not use the new keyword. `_cache_key` and diskcache storage formats do not change.

### Reading pre-1.7.0 replies

On a miss under the new key, `LLMClient.generate` looks up the key the same request had before
1.7.0: the same `_cache_key` over the old-layout text, with the same model, schema, sampling
parameters and namespace. A hit is returned as any cache hit, and no paid call is made.

- Read-only: nothing is written under the old key, and an old-layout hit is not copied to the
  new key. New replies are stored under the new key only.
- One small function renders the old order, used only to compute this key. The grader passes
  that text to `generate` alongside the new blocks, so other callers are unaffected.
- It applies only to per-criterion requests with text content. Image documents have no
  pre-1.7.0 form, so after the image sync they skip the lookup.
- Consequence: a run reading an old cache can return verdicts generated under the old order.
  The version note says so, and clearing the cache gives new-layout replies. The measurement
  uses a fresh cache, so it is unaffected.

### Warm-first scheduling

`_judge_all_criteria_for_judge` today gathers all of an item's criterion calls at once.
Under this design, when the prefix can be cached, it sends warming calls first and then
gathers the rest. Anthropic's entry only becomes available once the first response starts,
and the other providers' caches are similar in practice.

- Binary and multi-choice criteria use different system messages, so they share no prefix.
  Each system prompt gets its own warming call: the lowest-index criterion of each kind. The
  warming calls run in parallel with each other, so a mixed rubric still waits about one call.
- Applies per system prompt, when the judge has `LLMConfig.prompt_caching=True`, at least two
  calls share that system prompt, the model is known to cache, and the prefix reaches its
  minimum. A kind with one call needs no warming and joins the gather.
- A model is known to cache when `supports_prompt_caching` is true. Unknown models and models
  without caching skip warm-first and are not serialized.
- The minimum comes from `prompt_cache_min_tokens` in LiteLLM's model map. That field exists
  only for Anthropic-family entries, so Gemini-family models use a module constant of 4,096
  (Gemini's documented minimum for 3.x; 2.5 is lower, and the larger value only skips warming
  there) and other models use LiteLLM's fallback of 1,024. Recorded Gemini prompts therefore
  skip warm-first and stay unchanged in timing.
- Token count is an estimate (`litellm.token_counter` or similar); the estimate need not be
  exact, since a wrong one costs either a little latency or a miss.
- A warming call's failure does not stop the others. Errors route as they do today; a
  configuration error still raises and cancels as `_gather_or_cancel` does.
- The extra latency is about one call per item, per judge. Judges of a panel and escalation
  judges start in parallel as they do today, so same-model judges can miss together; this is
  accepted. Report the latency in the measurement.
- The same rule applies to a cascade's escalated per-criterion calls, which use the same path.

`LLMConfig.prompt_caching` now governs the split message, the breakpoints and warm-first
scheduling for every provider. Its docstring, which says Anthropic only, changes with it.
With `prompt_caching=False` the prompt keeps the new order, sent as one user message.

### Compatibility

- `ExperimentManifest` gains no key. Like `llm_calls`, the layout is not checked on resume, so
  resuming a pre-1.7.0 run mixes layouts. The version note says so.
- Reports, checkpoints, pickles and cache directories keep their formats. A test uses a
  response cache written by the old code path, pinned as a fixture, and confirms it loads and
  that a new-layout request for the same content hits it with no call. Goldens pin the
  old-layout text, so the fallback key cannot drift.
- `VERSION` stays 1.6.1 on this branch unless the release step asks otherwise; the note is
  written for 1.7.0.

### Sync with `autorubric-image`

That branch's `prompts.py` rewrites these builders to assemble content parts with `_concat`
and `_input_block`/`_reference_block`/`_submission_block`. Keep the builder changes small and
keep shared and tail as separately assembled pieces so the sync is mechanical, and image parts
land in the shared block. Do not touch that branch here.

## Tests

- Goldens for all four builders, with and without guidelines, query, reference and examples.
- The tail is the end of the prompt, and the criterion ends it in every builder.
- Two criteria of one item give identical shared blocks; so do all judges' calls with the same
  guidelines, and multi-choice and binary criteria on one item.
- `LLMClient` messages per provider: two user messages, Anthropic markers, routed Anthropic id,
  `prompt_caching=False`, callers without the keyword unchanged.
- Warm-first order: first call completes before the others start; gating off for one criterion,
  short prefix or `prompt_caching=False`; failure of the first call; a configuration error.
- Per-item, decision-model and meta prompts stay byte-identical (existing goldens).
- Old cache directories load and hit (above). The fallback writes nothing, a new-key entry wins
  over an old one, changed sampling parameters or namespace miss, and `use_cache=False` or a
  disabled cache skips it.

## Measurement

Per-item runs are already recorded, so only per-criterion re-runs. Follow the existing
harness (`jev_experiments/launch_per_item.sh`, `per_item_compare.py`) with a fresh response
cache and a new results root. Same items, models, temperatures and master seeds.

- Default scope: Luna and DeepSeek, about $6 from recorded costs. Gemini's run is about $20
  and is only worth it on the author's request, since nothing should change for it.
- Do a small smoke run first. Cost and scope are approved by the author before any paid call.
- Report per mode: recorded cost, cached prompt tokens, wall time, the old and new ratio to
  per-item, and verdict agreement with the recorded per-criterion runs, read against the
  run-to-run agreement baseline already recorded in the per-item repeat.
- Also report cache-write tokens. Luna charges 1.25x input for writes.
- The harness resolves `autorubric` from `..`, which is the main checkout, so it must be
  pointed at this branch for the run.

The measurement is a step before the PR is ready: the work stops for consent there. Cost
results are reported and do not gate the layout (Decision 1). Verdict agreement clearly below
the run-to-run baseline goes back to the author before merge.

Estimates before running, not results: Luna from about 1.8x to about 1.5x, DeepSeek a smaller
gain because output dominates, Gemini unchanged. Per-criterion output is about 1.4-1.5x
per-item, and caching cannot reduce that.

## Docs

- `dev-reference/`: `grading-flow.md`, `conventions.md`, `reproducibility.md`, `types.md`, and
  the `index.md` TLDRs for new names.
- User docs on prompt caching and a cost note in `docs/`, with measured numbers, including
  which providers cache what and Gemini's minimum.
- Version note: the new layout, old cached replies still reused and what that means for
  verdicts, resume mixing, measured agreement.

## Left to the builder

- The exact keyword name on `generate`, and how builders expose shared and tail while keeping
  their public return value a single joined string.
- Whether warm-first lives in `_judge_all_criteria_for_judge` or a small helper.
- Which LiteLLM helper estimates tokens.
- Updating the per-criterion assertion hard-coded in `tests/test_per_item_prompt_goldens.py`.
