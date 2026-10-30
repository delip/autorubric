# Per-criterion Prompt Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make default per-criterion grading reuse provider prompt caches by putting the item's shared content first and the criterion last, warming the cache before an item's parallel calls, while old response caches keep hitting.

**Architecture:** `prompts.py` splits each per-criterion prompt into a shared block and a tail. `LLMClient.generate` sends them as two user messages where the model caches prompts, and falls back to the pre-1.7.0 key on a response-cache miss. `CriterionGrader` sends one warming call per system prompt, then the rest.

**Tech Stack:** Python 3.11, LiteLLM, diskcache, pytest, `uv`, ruff, `ty`.

**Spec:** `docs/superpowers/specs/2026-10-30-per-criterion-prompt-caching-design.md`

## Global Constraints

- Prompts for `llm_calls="per_item"`, decision models, meta-rubric and revision stay byte-identical.
- Few-shot selection and content do not change, only position.
- Response cache, checkpoint, pickle and manifest formats do not change.
- Line length 100, async tests carry `@pytest.mark.asyncio`, `ty` for types, no new types unless needed.
- No paid API call without the author's consent for that run.
- No ephemeral task ids, no Claude bylines. Never touch `autorubric-image`.
- Gate on `uv run ruff check src/autorubric tests examples` and `uv run ruff format --diff`.

## Review Focus

- A strict OpenAI-compatible server that refuses consecutive user messages: a model not known to cache gets one user message, as today.
- `supports_prompt_caching` raising or returning junk for an unknown model id: treated as not caching, never a grading failure.
- A rubric of one criterion, or an escalation with one criterion: no warming, one call.
- A warming call that fails with a routed error: the other criteria are still judged.
- A cached response under both the new and the old key: the new one wins.

---

### Task 1: Prompt order

**Files:**
- Modify: `src/autorubric/prompts.py`
- Modify: `tests/golden/prompts/**` (regenerated), `tests/test_per_item_prompt_goldens.py`, `tests/test_prompts.py`
- Test: `tests/test_prompt_goldens.py`, `tests/test_prompts.py`

**Interfaces:**
- Produces, in `prompts.py`:
  - `_shared_user_content(to_grade: str, query: str | None, reference_submission: str | None, *, guidelines: str | None) -> str`
  - `_criterion_tail(criterion_text: str, examples_text: str = "") -> str`
  - `_legacy_user_prompt(criterion_text: str, examples_text: str, to_grade: str, query: str | None, reference_submission: str | None, *, guidelines: str | None) -> str`, the pre-1.7.0 text, used only for the fallback key
  - the four `build_*` functions still return one `str`: `f"{shared}\n\n{tail}"`

- [ ] **Step 1: Write failing tests** `test_criterion_ends_every_per_criterion_prompt`, `test_examples_sit_just_before_the_criterion`, `test_shared_block_is_equal_across_criteria_of_an_item`, `test_legacy_prompt_is_the_old_layout` (literal strings, including guidelines, query, reference, examples).
- [ ] **Step 2:** Run `uv run pytest tests/test_prompts.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the helpers and rewrite the four builders on them.
- [ ] **Step 4:** Regenerate per-criterion goldens, review the diff shows reordering only, and fix the hard-coded assertion in `tests/test_per_item_prompt_goldens.py`.
- [ ] **Step 5:** Run `uv run pytest tests/test_prompts.py tests/test_prompt_goldens.py tests/test_per_item_prompt_goldens.py tests/meta`. Expected: PASS, with per-item and meta goldens unchanged.
- [ ] **Step 6:** Commit.

### Task 2: `LLMClient` split messages and cache fallback

**Files:**
- Modify: `src/autorubric/llm.py`
- Test: `tests/llm/test_prompt_caching.py` (create)

**Interfaces:**
- Produces, in `llm.py`:
  - `_is_anthropic_model(model: str) -> bool`: `anthropic/`, `claude`, `openrouter/anthropic/`
  - `_caches_prompts(model: str) -> bool`: `litellm.supports_prompt_caching` true, any exception false, Bedrock and Vertex Claude ids false
  - `_prompt_cache_min_tokens(model: str) -> int`: map's `prompt_cache_min_tokens`, then `_GEMINI_PROMPT_CACHE_MIN_TOKENS = 4096` for Gemini-family ids, then `litellm.constants.DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT`
  - `LLMClient.generate(..., return_result=False, shared_prompt: str | None = None, legacy_user_prompt: str | None = None, **kwargs)`. With `shared_prompt`, `user_prompt` is the tail.

Message rules: with `shared_prompt` set, `prompt_caching=True` and `_caches_prompts(model)`, send `[system, user(shared), user(tail)]`. Otherwise send one user message `f"{shared}\n\n{tail}"`. Anthropic models get `cache_control` on the system message and on the shared block.

- [ ] **Step 1: Write failing tests** on the `litellm.acompletion` call's `messages`: `test_split_for_a_caching_model`, `test_breakpoints_on_anthropic`, `test_openrouter_anthropic_is_marked`, `test_one_message_for_a_model_not_known_to_cache`, `test_one_message_without_prompt_caching`, `test_bedrock_claude_is_unmarked_and_joined`, `test_callers_without_shared_prompt_unchanged`, `test_lookup_failure_means_not_caching`, `test_min_tokens_gemini_and_fallback`.
- [ ] **Step 2: Write failing cache tests:** `test_key_hashes_the_joined_text`, `test_old_cached_reply_hits_with_no_call` (entry written under a SHA-256 pinned from `main`'s `_cache_key`), `test_new_entry_wins_over_old`, `test_fallback_writes_nothing`, `test_changed_temperature_or_namespace_misses`, `test_use_cache_false_skips_fallback`.
- [ ] **Step 3:** Run `uv run pytest tests/llm/test_prompt_caching.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement. Update the `prompt_caching` docstring and the comments in `generate`.
- [ ] **Step 5:** Run `uv run pytest tests/llm`. Expected: PASS.
- [ ] **Step 6:** Commit.

### Task 3: Grader wiring and warm-first

**Files:**
- Modify: `src/autorubric/graders/criterion_grader.py`
- Test: `tests/graders/test_warm_first.py` (create)

**Interfaces:**
- Consumes: Task 1 helpers, Task 2 `generate` keywords, `_caches_prompts`, `_prompt_cache_min_tokens`.
- Produces: `CriterionGrader._judge_criteria(self, judge: JudgeSpec, criteria: list[tuple[int, Criterion]], to_grade, query, reference_submission, *, guidelines) -> list[CriterionResult]`, results in `criteria` order, called from `_judge_all_criteria_for_judge` and `_judge_escalated_criteria`.

Rule: group `criteria` by `criterion.is_multi_choice`. A group with at least two calls warms with its lowest-index member when `judge.llm_config.prompt_caching`, `_caches_prompts(model)`, and `litellm.token_counter` of system prompt plus shared block reaches `_prompt_cache_min_tokens(model)`. Warming calls run together through `_gather_or_cancel`, then the rest.

- [ ] **Step 1: Write failing tests:** `test_warming_call_finishes_before_the_rest`, `test_mixed_rubric_warms_each_kind_in_parallel`, `test_lone_criterion_of_a_kind_joins_the_gather`, `test_no_warming_for_one_criterion`, `test_no_warming_when_prefix_is_short`, `test_no_warming_for_a_model_not_known_to_cache`, `test_no_warming_without_prompt_caching`, `test_failed_warming_call_leaves_the_rest_judged`, `test_configuration_error_raises`, `test_escalated_calls_warm_too`, `test_results_stay_in_rubric_order`, and `test_shuffle_permutation_unchanged`.
- [ ] **Step 2:** Run `uv run pytest tests/graders/test_warm_first.py -v`. Expected: FAIL.
- [ ] **Step 3:** Pass `shared_prompt` and `legacy_user_prompt` from `_judge_binary_criterion` and `_judge_multi_choice_criterion`, and implement `_judge_criteria`.
- [ ] **Step 4:** Run `uv run pytest`, then compare the test count with `main`. Expected: all pass, no test removed.
- [ ] **Step 5:** Run `uv run ty check` and compare diagnostics with `main`. Commit.

### Task 4: Docs

**Files:**
- Modify: `dev-reference/grading-flow.md`, `dev-reference/conventions.md`, `dev-reference/reproducibility.md`, `dev-reference/types.md`, `dev-reference/index.md`
- Modify: the user docs on grading cost and the changelog, wherever `docs/` keeps them

- [ ] **Step 1:** Update dev-reference pages and index keywords with every new name.
- [ ] **Step 2:** Write the user docs and version note, with measured numbers left for Task 5.
- [ ] **Step 3:** Run `uv run mkdocs build --strict` after `uv sync --group docs`. Expected: success. Commit.

### Task 5: Measurement

**Files:**
- Create: `jev_experiments/launch_prompt_cache.sh` (untracked directory, not committed on this branch)

- [ ] **Step 1:** Point the harness at this worktree's `autorubric`, with a new results root and a fresh response cache.
- [ ] **Step 2:** Stop and ask the author: smoke run scope and estimate.
- [ ] **Step 3:** After consent, run the smoke, check cached tokens rise, then ask again for the full run.
- [ ] **Step 4:** Report cost, cached and cache-write tokens, wall time, ratio to per-item, and verdict agreement against the recorded repeat baseline.
- [ ] **Step 5:** Put measured numbers in the docs and version note. Commit. A clear agreement drop goes back to the author before merge.

### Task 6: Finish

- [ ] **Step 1:** One fresh reviewer on the most capable model reads the whole branch diff.
- [ ] **Step 2:** Fix findings, rerun `uv run pytest`, ruff and `ty`.
- [ ] **Step 3:** Leave the branch ready for a PR.
