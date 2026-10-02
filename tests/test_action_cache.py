import asyncio
import json
from types import SimpleNamespace

from optexity.inference.core.interaction.action_cache import (
    ActionCacheRecorder,
    classify_actions,
)
from optexity.schema.action_cache import ActionCache


def classify(*actions):
    cache = ActionCache(task="test", actions=list(actions))
    classify_actions(cache)
    return [(a.category, a.reason) for a in cache.actions]


class TestClassifyActions:
    def test_unique_locator_is_deterministic(self, make_action):
        assert classify(make_action("input", {"text": "SF"})) == [
            ("deterministic", "unique role+name locator")
        ]

    def test_ambiguous_locator_needs_llm(self, make_action):
        [(category, _)] = classify(make_action("input", {"text": "SF"}, unique=False))
        assert category == "needs_llm"

    def test_failed_done_and_exploratory_actions_are_redundant(self, make_action):
        results = classify(
            make_action("click", error="element not found"),
            make_action("scroll", {"down": True}),
            make_action("done", {"text": "ok", "success": True}),
        )
        assert [category for category, _ in results] == ["redundant"] * 3

    def test_input_overwritten_later_is_redundant(self, make_action):
        results = classify(
            make_action("input", {"text": "first"}, step=1),
            make_action("input", {"text": "second"}, step=2),
        )
        assert results[0] == ("redundant", "overwritten by action 1")
        assert results[1][0] == "deterministic"

    def test_failed_input_does_not_overwrite_earlier_one(self, make_action):
        results = classify(
            make_action("input", {"text": "kept"}, step=1),
            make_action("input", {"text": "lost"}, step=2, error="detached"),
        )
        assert [category for category, _ in results] == ["deterministic", "redundant"]

    def test_inputs_on_different_elements_are_both_kept(self, make_action):
        results = classify(
            make_action("input", {"text": "a"}, xpath="html/body/form/input[1]"),
            make_action("input", {"text": "b"}, xpath="html/body/form/input[2]"),
        )
        assert [category for category, _ in results] == ["deterministic"] * 2

    def test_navigate_to_the_open_page_is_redundant(self, make_action):
        same, other = classify(
            make_action("navigate", {"url": "https://example.com/form"}),
            make_action("navigate", {"url": "https://example.com/other"}),
        )
        assert same == ("redundant", "already on that page")
        assert other[0] == "deterministic"

    def test_page_reading_and_tab_actions_need_llm(self, make_action):
        results = classify(
            make_action("extract", {"query": "price"}),
            make_action("switch", {"tab_id": "ab12"}),
        )
        assert [category for category, _ in results] == ["needs_llm"] * 2


class _Action:
    """Stands in for browser-use's ``ActionModel``."""

    def __init__(self, name: str, params: dict):
        self._data = {name: params}

    def model_dump(self, exclude_unset: bool = False) -> dict:
        return self._data


def _event(step: int, name: str, params: dict, result=None):
    return SimpleNamespace(
        step=step,
        action_index=0,
        action=_Action(name, params),
        element=None,
        target_id=None,
        result=result,
    )


def _result(**overrides):
    fields = {
        "error": None,
        "is_done": False,
        "success": None,
        "extracted_content": None,
    }
    return SimpleNamespace(**{**fields, **overrides})


class TestActionCacheRecorder:
    def _recorder(self, downloads_dir):
        browser = SimpleNamespace(
            memory=SimpleNamespace(urls_to_downloads=[], raw_downloads={}),
            temp_downloads_dir=str(downloads_dir),
            context=None,
        )
        return ActionCacheRecorder(browser, task="download the report")

    def _run(self, recorder, step, name, params, result=None):
        asyncio.run(recorder.on_action_start(_event(step, name, params)))
        asyncio.run(recorder.on_action_end(_event(step, name, params, result)))

    def test_records_start_and_end_as_one_action(self, tmp_path):
        recorder = self._recorder(tmp_path)
        self._run(recorder, 1, "navigate", {"url": "https://example.com"}, _result())

        [action] = recorder.cache.actions
        assert action.action_name == "navigate"
        assert action.params == {"url": "https://example.com"}
        assert action.duration_s is not None

    def test_download_landing_after_click_is_credited_to_that_click(self, tmp_path):
        recorder = self._recorder(tmp_path)
        self._run(recorder, 1, "click", {"index": 4}, _result())
        (tmp_path / "report.csv").write_text("a,b\n")
        self._run(recorder, 2, "click", {"index": 9}, _result())

        first, second = recorder.cache.actions
        assert first.started_download is True
        assert second.started_download is False

    def test_save_credits_a_download_that_lands_after_the_last_action(self, tmp_path):
        downloads = tmp_path / "downloads"
        downloads.mkdir()
        recorder = self._recorder(downloads)
        self._run(recorder, 1, "click", {"index": 4}, _result())
        (downloads / "report.csv").write_text("a,b\n")

        recorder.save(tmp_path / "action_cache.json")

        [click] = json.loads((tmp_path / "action_cache.json").read_text())["actions"]
        assert click["started_download"] is True
        assert click["category"] is not None
