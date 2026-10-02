import json

import pytest

from optexity.inference.core.interaction import llm_automation_builder
from optexity.inference.core.interaction.action_cache import classify_actions
from optexity.inference.core.interaction.cached_automation import compile_action_cache
from optexity.inference.core.interaction.llm_automation_builder import (
    _check,
    build_cached_automation_with_llm,
)
from optexity.schema.action_cache import ActionCache
from optexity.schema.automation import Automation
from optexity.schema.token_usage import TokenUsage


@pytest.fixture
def cache(make_action) -> ActionCache:
    """A run that typed a city and downloaded a report."""
    cache = ActionCache(
        task="type SF as the city, then download the report",
        actions=[
            make_action("input", {"text": "SF"}, step=1),
            make_action(
                "click",
                {"index": 7},
                step=2,
                locator='get_by_role("link", name="CSV")',
                xpath="html/body/a[1]",
                started_download=True,
            ),
            make_action("done", {"text": "done"}, step=3),
        ],
    )
    classify_actions(cache)
    return cache


@pytest.fixture
def draft(cache) -> list[dict]:
    """The code builder's nodes, as the LLM would echo them back."""
    return [n.model_dump(exclude_defaults=True) for n in compile_action_cache(cache)]


def answer(nodes: list[dict], params: dict | None = None) -> str:
    return json.dumps({"input_parameters": params or {}, "nodes": nodes})


def errors_for(text: str, cache: ActionCache) -> list[str]:
    _, errors = _check(text, cache, existing_params={})
    return errors


class TestCheck:
    def test_accepts_the_code_draft(self, cache, draft):
        response, errors = _check(answer(draft), cache, existing_params={})
        assert errors == []
        assert len(response.nodes) == 2

    def test_rejects_non_json(self, cache):
        assert errors_for("Sure! Here is the automation.", cache) == [
            "Response is not the requested JSON object."
        ]

    def test_rejects_a_locator_the_cache_never_verified(self, cache, draft):
        draft[0]["interaction_action"]["input_text"]["command"] = 'locator("#city")'
        assert any("verified locators" in e for e in errors_for(answer(draft), cache))

    def test_rejects_text_the_agent_never_typed(self, cache, draft):
        draft[0]["interaction_action"]["input_text"]["input_text"] = "LA"
        assert any("never typed" in e for e in errors_for(answer(draft), cache))

    def test_accepts_a_parameter_defaulting_to_the_typed_value(self, cache, draft):
        draft[0]["interaction_action"]["input_text"]["input_text"] = "{city[0]}"
        assert errors_for(answer(draft, {"city": ["SF"]}), cache) == []

    def test_rejects_a_parameter_with_a_different_value(self, cache, draft):
        draft[0]["interaction_action"]["input_text"]["input_text"] = "{city[0]}"
        errors = errors_for(answer(draft, {"city": ["LA"]}), cache)
        assert any("never typed" in e for e in errors)

    def test_rejects_undeclared_and_unused_parameters(self, cache, draft):
        draft[0]["interaction_action"]["input_text"]["input_text"] = "{town[0]}"
        errors = errors_for(answer(draft, {"unused": ["x"]}), cache)
        assert any("'{town[...]}' is not declared" in e for e in errors)
        assert any("'unused' is declared but never used" in e for e in errors)

    def test_rejects_a_missing_download(self, cache, draft):
        draft[1]["interaction_action"]["click_element"]["expect_download"] = False
        errors = errors_for(answer(draft), cache)
        assert any("expect_download" in e for e in errors)

    def test_rejects_a_disallowed_interaction(self, cache, draft):
        draft[0] = {
            "type": "action_node",
            "interaction_action": {
                "agentic_task": {"task": "x", "max_steps": 3, "backend": "browser_use"}
            },
        }
        assert any("must be one of" in e for e in errors_for(answer(draft), cache))


class _ScriptedModel:
    """Returns the given answers in order, like an LLM that corrects itself."""

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def get_model_response(self, prompt, system_instruction=None):
        self.prompts.append(prompt)
        return self.answers.pop(0), TokenUsage(total_tokens=100)


@pytest.fixture
def automation_with_cache(cache, tmp_path):
    (tmp_path / "step_0").mkdir()
    (tmp_path / "step_0" / "action_cache.json").write_text(cache.model_dump_json())
    automation = Automation.model_validate(
        {
            "url": "https://example.com/form",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [
                {
                    "type": "action_node",
                    "interaction_action": {
                        "agentic_task": {
                            "task": cache.task,
                            "max_steps": 5,
                            "backend": "browser_use",
                        }
                    },
                }
            ],
        }
    )
    return automation, tmp_path


def _use_model(monkeypatch, model: _ScriptedModel) -> None:
    monkeypatch.setattr(
        llm_automation_builder, "get_llm_model", lambda *args, **kwargs: model
    )


class TestBuildWithLLM:
    def test_retries_with_the_errors_until_the_answer_passes(
        self, monkeypatch, automation_with_cache, draft
    ):
        automation, logs = automation_with_cache
        invented = json.loads(json.dumps(draft))
        invented[0]["interaction_action"]["input_text"]["command"] = 'locator("#x")'
        draft[0]["interaction_action"]["input_text"]["input_text"] = "{city[0]}"
        model = _ScriptedModel([answer(invented), answer(draft, {"city": ["SF"]})])
        _use_model(monkeypatch, model)

        built, report = build_cached_automation_with_llm(automation, logs, "fake")

        assert [len(a.errors) > 0 for a in report.attempts] == [True, False]
        assert "verified locators" in model.prompts[1]
        assert built.parameters.input_parameters == {"city": ["SF"]}
        assert built.nodes[0].interaction_action.input_text.input_text == "{city[0]}"
        assert built.expected_downloads == 1
        assert report.tokens.total_tokens == 200

    def test_keeps_the_agentic_task_when_every_attempt_fails(
        self, monkeypatch, automation_with_cache
    ):
        automation, logs = automation_with_cache
        _use_model(monkeypatch, _ScriptedModel(["not json"] * 3))

        built, report = build_cached_automation_with_llm(
            automation, logs, "fake", max_attempts=3
        )

        assert report.kept_agentic == [0]
        assert built.nodes[0].interaction_action.agentic_task is not None
