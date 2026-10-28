"""Grading submissions that hold images, through ``Grader.grade`` and ``CriterionGrader``.

``Grader.grade`` puts the submission, query and reference submission in canonical form and
reads their image files once, off the event loop, before any judge is called; a missing or
unreadable image raises from the grade. Every LLM judge (a single judge, either way it is
called, each member of a panel, a cascade's escalation judge) receives the loaded document
and sends its images as content parts. Few-shot examples whose submissions hold images are
read once at construction and shown as parts too. The length penalty counts text only.

Nothing here reaches the network: the real ``LLMClient`` is kept and ``litellm.acompletion``
is patched by a recorder that answers each call from its response format. Decision-model
requests go to the recording fake SDK client of the decision-model tests.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import PIL.Image as pil_image
import pytest

# The decision-model tests' guards and fake SDK, registered for this module by importing
# them: no ``httpx2`` transport may send, the ``TYPESAFE_*`` variables are cleared, and the
# rate-limit pool starts fresh for each test.
from decision.conftest import (  # noqa: F401
    FakeSDK,
    _fresh_rate_limit_pool,
    _isolate_typesafe_env,
    _no_network,
    fake_sdk,
    make_response,
)
from litellm import ModelResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DataItem,
    DecisionModelConfig,
    EscalationConfig,
    FewShotConfig,
    Image,
    LengthPenalty,
    LLMConfig,
    Rubric,
    RubricDataset,
)
from autorubric import content as content_module
from autorubric.content import _content_digest, _normalize_content
from autorubric.graders import CriterionGrader, JudgeSpec

SEED = 5
API_KEY = "ts-test-key-never-persisted-5a1d"
CAPTION = "Figure 1: accuracy by model size."
TEXT = "Accuracy grows with model size."

LEGIBLE = Criterion(name="legible", weight=2.0, requirement="The figure is legible")
CLUTTERED = Criterion(name="cluttered", weight=-1.0, requirement="The figure is cluttered")
CLARITY = Criterion(
    name="clarity",
    weight=1.0,
    requirement="How clear is the figure?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Unclear", value=0.0),
        CriterionOption(label="Somewhat clear", value=0.4),
        CriterionOption(label="Clear", value=0.7),
        CriterionOption(label="Very clear", value=1.0),
    ],
)
MIXED = [LEGIBLE, CLARITY, CLUTTERED]
MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET


def encoded_png(color: str) -> bytes:
    """A tiny PNG of one ``color``, encoded by Pillow."""
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


def data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def png_file(directory: Path, name: str, color: str) -> Path:
    path = directory / name
    path.write_bytes(encoded_png(color))
    return path


def llm() -> LLMConfig:
    return LLMConfig(model="openai/gpt-4.1-mini")


def model_response(content: str) -> ModelResponse:
    return ModelResponse(
        model="gpt-4.1-mini",
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


class Provider:
    """``litellm.acompletion``'s stand-in: records each call's parameters and answers it
    from its response format (every binary criterion MET, every multi-choice criterion its
    first option shown)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, **params: Any) -> ModelResponse:
        self.calls.append(params)
        name = params["response_format"]["json_schema"]["name"]
        if name == "CriterionJudgment":
            reply: dict[str, Any] = {"criterion_status": "MET", "explanation": "Legible."}
        elif name == "MultiChoiceJudgment":
            reply = {"selected_option": 1, "explanation": "Clear enough."}
        else:
            assert name == "RubricJudgment", name
            ids = re.findall(r'<rubric_criterion id="(c\d+)">', user_text(params))
            reply = {
                "judgments": [
                    {
                        "criterion_id": criterion_id,
                        "criterion_status": "MET",
                        "selected_option": 1,
                        "explanation": "Fine.",
                    }
                    for criterion_id in ids
                ]
            }
        return model_response(json.dumps(reply))

    def image_urls(self) -> list[list[str]]:
        """Each call's image parts' URLs, in order."""
        return [
            [part["image_url"]["url"] for part in user_parts(params) if part["type"] == "image_url"]
            for params in self.calls
        ]


def user_message(params: dict[str, Any]) -> Any:
    (message,) = [m for m in params["messages"] if m["role"] == "user"]
    return message["content"]


def user_parts(params: dict[str, Any]) -> list[dict[str, Any]]:
    content = user_message(params)
    assert isinstance(content, list), "a prompt holding images is sent as content parts"
    return content


def user_text(params: dict[str, Any]) -> str:
    content = user_message(params)
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part["type"] == "text")


@pytest.fixture
def provider() -> Iterator[Provider]:
    recorder = Provider()
    with patch("litellm.acompletion", side_effect=recorder.acompletion):
        yield recorder


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> dict[Path, int]:
    """How many times each file was read with ``Path.read_bytes``."""
    counts: dict[Path, int] = {}
    real = Path.read_bytes

    def counting(self: Path) -> bytes:
        counts[self] = counts.get(self, 0) + 1
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    return counts


def figure(tmp_path: Path, color: str = "red") -> tuple[list[str | Image], Path]:
    """A document holding a figure read from a file, then its caption."""
    path = png_file(tmp_path, f"figure-{color}.png", color)
    return [Image(path), CAPTION], path


# =============================================================================
# The images reach every LLM judge's call
# =============================================================================


class TestImagesReachTheProvider:
    @pytest.mark.asyncio
    async def test_per_criterion_calls_each_send_the_figure(self, tmp_path, provider):
        document, path = figure(tmp_path)
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        report = await Rubric(MIXED).grade(document, grader=grader)

        assert report.error is None
        expected = data_url(path.read_bytes())
        assert provider.image_urls() == [[expected]] * len(MIXED)
        for params in provider.calls:
            parts = user_parts(params)
            image_at = next(i for i, part in enumerate(parts) if part["type"] == "image_url")
            # Spliced into the submission block, right after its opening tag.
            assert parts[image_at - 1]["text"].endswith("<submission>\n")
            assert parts[image_at + 1]["text"].startswith(CAPTION)

    @pytest.mark.asyncio
    async def test_a_per_item_call_sends_the_figure_once(self, tmp_path, provider):
        document, path = figure(tmp_path)
        grader = CriterionGrader(judge_model_config=llm(), llm_calls="per_item", seed=SEED)
        report = await Rubric(MIXED).grade(document, grader=grader)

        assert report.error is None
        assert provider.image_urls() == [[data_url(path.read_bytes())]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_every_panel_member_sends_the_figure(self, tmp_path, provider, llm_calls):
        document, path = figure(tmp_path)
        grader = CriterionGrader(
            judges=[JudgeSpec(llm(), judge_id="a"), JudgeSpec(llm(), judge_id="b")],
            llm_calls=llm_calls,
            seed=SEED,
        )
        await Rubric(MIXED).grade(document, grader=grader)

        calls_per_judge = len(MIXED) if llm_calls == "per_criterion" else 1
        assert provider.image_urls() == [[data_url(path.read_bytes())]] * (2 * calls_per_judge)

    @pytest.mark.asyncio
    async def test_query_and_reference_images_are_sent_too(self, tmp_path, provider):
        document, path = figure(tmp_path)
        query_image = encoded_png("green")
        reference = png_file(tmp_path, "reference.png", "blue")
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        await Rubric([LEGIBLE]).grade(
            document,
            grader=grader,
            query=[Image(query_image), "Rate this figure."],
            reference_submission=[Image(reference), "A model figure."],
        )

        (urls,) = provider.image_urls()
        assert sorted(urls) == sorted(
            [data_url(query_image), data_url(reference.read_bytes()), data_url(path.read_bytes())]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_a_cascade_sends_the_loaded_figure_to_both_stages(
        self,
        tmp_path,
        provider,
        reads,
        llm_calls,
        fake_sdk: FakeSDK,  # noqa: F811
    ):
        """The decision model and the escalation judge go through the same ``judge`` path,
        so both receive the content ``Grader.grade`` loaded, read once: the decision model
        sends the figure as its request's asset, and, unsure of every criterion (P = 0.5,
        confidence 0), escalates them all to the LLM, which sends it as a content part."""
        fake_sdk.response = make_response(
            {"c0": {"type": "noul", "noul": 0.5}, "c1": {"type": "noul", "noul": 0.5}}
        )
        document, path = figure(tmp_path)
        grader = CriterionGrader(
            judge_model_config=DecisionModelConfig(model="jev-latest", api_key=API_KEY),
            escalation=EscalationConfig(judges=llm(), threshold=0.5),
            llm_calls=llm_calls,
            seed=SEED,
        )

        rubric = [LEGIBLE, CLUTTERED]
        report = await Rubric(rubric).grade(document, grader=grader)

        assert report.report is not None
        assert all(criterion.escalated for criterion in report.report)
        calls = len(rubric) if llm_calls == "per_criterion" else 1
        assert provider.image_urls() == [[data_url(encoded_png("red"))]] * calls
        ((state, _, kwargs),) = fake_sdk.calls
        reference = f"<image:{Image(encoded_png('red')).digest}>"
        assert state == {"submission": reference + CAPTION}
        assert list(kwargs["assets"]) == [reference]
        assert reads[path] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("where", ["submission", "query", "reference submission"])
    async def test_a_decision_model_sends_an_image_wherever_it_is(
        self,
        fake_sdk: FakeSDK,  # noqa: F811
        where,
    ):
        grader = CriterionGrader(
            judges=[JudgeSpec(DecisionModelConfig(model="jev-latest", api_key=API_KEY), "jev")],
            seed=SEED,
        )
        image = Image(encoded_png("red"))
        kwargs: dict[str, Any] = {"to_grade": TEXT}
        kwargs[{"submission": "to_grade", "query": "query"}.get(where, "reference_submission")] = [
            image,
            CAPTION,
        ]
        await Rubric([LEGIBLE]).grade(grader=grader, **kwargs)

        ((state, _, sent),) = fake_sdk.calls
        reference = f"<image:{image.digest}>"
        field = {"submission": "submission", "query": "input"}.get(where, "reference_submission")
        assert state[field] == reference + CAPTION
        assert list(sent["assets"]) == [reference]
        assert sent["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}


# =============================================================================
# Loading: once per grade, before any judge, raising on a bad image
# =============================================================================


class TestLoading:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("where", ["submission", "query", "reference_submission"])
    async def test_a_missing_file_raises_before_any_call(self, tmp_path, provider, where):
        missing: list[str | Image] = [Image(tmp_path / "missing.png"), CAPTION]
        kwargs: dict[str, Any] = {"to_grade": TEXT}
        kwargs["to_grade" if where == "submission" else where] = missing
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)

        with pytest.raises(FileNotFoundError):
            await Rubric(MIXED).grade(grader=grader, **kwargs)
        assert provider.calls == []

    @pytest.mark.asyncio
    async def test_a_file_that_is_not_an_image_raises_before_any_call(self, tmp_path, provider):
        path = tmp_path / "notes.png"
        path.write_text("not an image")
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)

        with pytest.raises(ValueError, match="not an image Pillow can identify"):
            await Rubric(MIXED).grade([Image(path), CAPTION], grader=grader)
        assert provider.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_each_file_is_read_once_per_grade(self, tmp_path, provider, reads, llm_calls):
        document, path = figure(tmp_path)
        reference = png_file(tmp_path, "reference.png", "blue")
        grader = CriterionGrader(
            judges=[JudgeSpec(llm(), judge_id="a"), JudgeSpec(llm(), judge_id="b")],
            llm_calls=llm_calls,
            seed=SEED,
        )
        for graded in range(1, 3):
            await Rubric(MIXED).grade(
                document, grader=grader, reference_submission=[Image(reference)]
            )
            assert (reads[path], reads[reference]) == (graded, graded)

    @pytest.mark.asyncio
    async def test_a_direct_judge_call_loads_the_content_itself(self, tmp_path, provider, reads):
        document, path = figure(tmp_path)
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        await grader.judge(document, MIXED)

        assert reads[path] == 1
        assert provider.image_urls() == [[data_url(encoded_png("red"))]] * len(MIXED)

    @pytest.mark.asyncio
    async def test_files_are_read_in_a_worker_thread_and_nothing_else_is(
        self, tmp_path, provider, monkeypatch
    ):
        threaded: list[Any] = []
        real = asyncio.to_thread

        async def recording(function: Any, *args: Any) -> Any:
            threaded.append(args)
            return await real(function, *args)

        monkeypatch.setattr(content_module.asyncio, "to_thread", recording)
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        rubric = Rubric([LEGIBLE])

        await rubric.grade(TEXT, grader=grader, query="Rate it.", reference_submission="Ref.")
        await rubric.grade([Image(encoded_png("red")), CAPTION], grader=grader)
        assert threaded == []

        document, _path = figure(tmp_path)
        await rubric.grade(document, grader=grader)
        assert threaded == [(_normalize_content(document),)]

    @pytest.mark.asyncio
    async def test_text_reaches_the_judge_as_the_same_string(self, provider):
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        received: list[Any] = []
        judge = grader.judge

        async def recording(to_grade: Any, *args: Any, **kwargs: Any) -> Any:
            received.append(to_grade)
            return await judge(to_grade, *args, **kwargs)

        with patch.object(grader, "judge", recording):
            await Rubric([LEGIBLE]).grade(TEXT, grader=grader)
        (passed,) = received
        assert passed is TEXT
        assert user_message(provider.calls[0]).count(TEXT) == 1


# =============================================================================
# What the grade keys on a document, and what it refuses
# =============================================================================


class TestShuffleKey:
    @pytest.mark.asyncio
    async def test_a_documents_shuffle_order_is_the_one_its_digest_gives(self, tmp_path, provider):
        document, path = figure(tmp_path)
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        (results,) = await grader.judge(document, [CLARITY])
        (result,) = results.criterion_results
        assert result is not None
        recorded = result.report.shuffle_order
        assert recorded is not None

        # The same content held as bytes has the same digest, so the same permutation.
        same_content = [Image(path.read_bytes()), CAPTION]
        assert _content_digest(same_content) == _content_digest(document)
        effective = grader._effective_criterion(CLARITY)
        _shown, order = grader._presented_criterion(effective, 0, "default", same_content)
        assert recorded == order
        assert recorded != sorted(recorded)  # shuffled, not left in order


class TestRefusals:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "value", [Image(encoded_png("red")), [Image(encoded_png("red")), CAPTION], ["a", "b"]]
    )
    @pytest.mark.parametrize("key", ["thinking", "output"])
    async def test_a_thinking_output_dict_holding_images_is_refused(self, provider, key, value):
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        submission = cast(Any, {key: value})
        with pytest.raises(TypeError, match=f"The '{key}' section"):
            await Rubric([LEGIBLE]).grade(submission, grader=grader)
        assert provider.calls == []

    @pytest.mark.asyncio
    async def test_a_text_thinking_output_dict_is_flattened_as_before(self, provider):
        grader = CriterionGrader(judge_model_config=llm(), seed=SEED)
        submission = cast(Any, {"thinking": "Look at the axes.", "output": "Legible."})
        await Rubric([LEGIBLE]).grade(submission, grader=grader)
        text = user_text(provider.calls[0])
        assert "<thinking>Look at the axes.</thinking>\n<output>Legible.</output>" in text


# =============================================================================
# The length penalty counts text only
# =============================================================================


class TestLengthPenalty:
    @pytest.mark.asyncio
    async def test_a_document_is_counted_by_its_words(self, tmp_path, provider):
        counted: list[str] = []

        def words(text: str) -> int:
            counted.append(text)
            return len(text.split())

        grader = CriterionGrader(
            judge_model_config=llm(),
            seed=SEED,
            length_penalty=LengthPenalty(
                free_budget=3, max_cap=5, penalty_at_cap=0.5, exponent=1.0, count_fn=words
            ),
        )
        document = ["words here", Image(encoded_png("red")), "more words"]
        report = await Rubric([LEGIBLE]).grade(document, grader=grader)

        assert counted == [" words here\nmore words"]
        assert report.score == pytest.approx(1.0 - 0.5 * (4 - 3) / (5 - 3))


# =============================================================================
# Few-shot examples whose submissions hold images
# =============================================================================


def training(
    tmp_path: Path,
    rubric: list[Criterion],
    labels: list[list[CriterionVerdict | str]],
    colors: list[str],
) -> tuple[RubricDataset, list[Path]]:
    """A training set: item ``i`` shows a figure of ``colors[i]`` (read from a file) with
    a caption, labelled ``labels[i]``."""
    dataset = RubricDataset(prompt="Rate the figure.", rubric=Rubric(rubric))
    paths = []
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        path = png_file(tmp_path, f"train-{i}.png", color)
        paths.append(path)
        submission = cast(Any, [Image(path), f"Training caption {i}."])
        dataset.items.append(
            DataItem(submission=submission, description=str(i), ground_truth=label)
        )
    return dataset, paths


class TestFewShotImages:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rubric, labels, llm_calls",
        [
            ([LEGIBLE], [[MET], [UNMET]], "per_criterion"),
            ([CLARITY], [["Clear"], ["Unclear"]], "per_criterion"),
            ([LEGIBLE, CLARITY], [[MET, "Clear"], [UNMET, "Unclear"]], "per_item"),
        ],
        ids=["binary", "multi_choice", "per_item"],
    )
    async def test_example_images_are_sent_as_parts(
        self, tmp_path, provider, reads, rubric, labels, llm_calls
    ):
        data, paths = training(tmp_path, rubric, labels, ["green", "blue"])
        grader = CriterionGrader(
            judges=[JudgeSpec(llm(), judge_id="a"), JudgeSpec(llm(), judge_id="b")],
            training_data=data,
            few_shot_config=FewShotConfig(n_examples=2),
            llm_calls=llm_calls,
            seed=SEED,
        )
        # Read once at construction, for both judges and every criterion.
        assert {path: reads[path] for path in paths} == dict.fromkeys(paths, 1)

        document, path = figure(tmp_path)
        await Rubric(rubric).grade(document, grader=grader)

        examples = {data_url(encoded_png("green")), data_url(encoded_png("blue"))}
        for urls in provider.image_urls():
            # Both examples' figures, then the graded figure, each once.
            assert sorted(urls[:2]) == sorted(examples)
            assert urls[2:] == [data_url(encoded_png("red"))]
        # Grading read the graded figure alone: the examples hold their bytes.
        assert {p: reads[p] for p in [*paths, path]} == {**dict.fromkeys(paths, 1), path: 1}

    @pytest.mark.asyncio
    async def test_the_same_figure_from_two_files_is_one_example(self, tmp_path, provider):
        data, paths = training(
            tmp_path, [LEGIBLE], [[MET], [MET], [UNMET]], ["green", "green", "blue"]
        )
        # Items 0 and 1 hold the same bytes, read from different files, and the same text.
        data.items[1].submission = cast(Any, [Image(paths[1]), "Training caption 0."])
        grader = CriterionGrader(
            judge_model_config=llm(),
            training_data=data,
            few_shot_config=FewShotConfig(n_examples=5),
            llm_calls="per_item",
            seed=SEED,
        )
        (examples,) = grader._item_examples.values()
        assert len(examples) == 2

        await Rubric([LEGIBLE]).grade(TEXT, grader=grader)
        (urls,) = provider.image_urls()
        assert sorted(urls) == sorted(
            [data_url(encoded_png("green")), data_url(encoded_png("blue"))]
        )

    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    def test_a_missing_training_image_fails_construction(self, tmp_path, llm_calls):
        data, paths = training(tmp_path, [LEGIBLE], [[MET], [UNMET]], ["green", "blue"])
        paths[0].unlink()
        with pytest.raises(FileNotFoundError):
            CriterionGrader(
                judge_model_config=llm(),
                training_data=data,
                few_shot_config=FewShotConfig(n_examples=2),
                llm_calls=llm_calls,
                seed=SEED,
            )

    def test_examples_hold_their_bytes_and_the_memo_is_dropped(self, tmp_path):
        data, _paths = training(tmp_path, [LEGIBLE], [[MET], [UNMET]], ["green", "blue"])
        grader = CriterionGrader(
            judge_model_config=llm(),
            training_data=data,
            few_shot_config=FewShotConfig(n_examples=2),
            seed=SEED,
        )
        assert grader._loaded_example_submissions is None
        examples = grader._criterion_examples[(0, "default")]
        assert len(examples) == 2
        for example in examples:
            submission = example.submission
            assert isinstance(submission, tuple)
            images = [part for part in submission if isinstance(part, Image)]
            assert len(images) == 1 and images[0]._bytes_held
