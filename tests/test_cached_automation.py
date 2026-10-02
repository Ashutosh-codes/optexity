import pytest

from optexity.inference.core.interaction.action_cache import classify_actions
from optexity.inference.core.interaction.cached_automation import (
    CACHED_LOCATOR_MAX_TRIES,
    UncompilableAction,
    build_cached_automation,
    compile_action_cache,
)
from optexity.schema.action_cache import ActionCache
from optexity.schema.automation import Automation


def cache_of(*actions) -> ActionCache:
    cache = ActionCache(task="test", actions=list(actions))
    classify_actions(cache)
    return cache


def done(make_action, success: bool = True):
    return make_action("done", {"text": "finished"}, step=9, success=success)


class TestCompileActionCache:
    def test_input_uses_the_verified_locator_verbatim(self, make_action):
        [node] = compile_action_cache(
            cache_of(make_action("input", {"text": "SF"}), done(make_action))
        )

        action = node.interaction_action
        assert action.input_text.command == 'get_by_role("textbox", name="City")'
        assert action.input_text.input_text == "SF"
        assert action.input_text.prompt_instructions == 'the textbox "City"'
        assert action.max_tries == CACHED_LOCATOR_MAX_TRIES

    def test_same_page_action_skips_the_load_wait(self, make_action):
        [node] = compile_action_cache(
            cache_of(make_action("input", {"text": "SF"}), done(make_action))
        )
        assert node.end_sleep_time == 0.0

    def test_navigating_click_keeps_the_load_wait(self, make_action):
        click = make_action("click", url_after="https://example.com/next")
        [node] = compile_action_cache(cache_of(click, done(make_action)))
        assert "end_sleep_time" not in node.model_fields_set

    def test_redundant_actions_are_dropped(self, make_action):
        nodes = compile_action_cache(
            cache_of(
                make_action("input", {"text": "typo"}, step=1),
                make_action("scroll", {"down": True}, step=2),
                make_action("input", {"text": "SF"}, step=3),
                done(make_action),
            )
        )
        assert [n.interaction_action.input_text.input_text for n in nodes] == ["SF"]

    def test_download_click_expects_a_download(self, make_action):
        click = make_action("click", started_download=True)
        [node] = compile_action_cache(cache_of(click, done(make_action)))
        assert node.interaction_action.click_element.expect_download is True

    def test_element_without_unique_locator_falls_back_to_the_prompt(self, make_action):
        ambiguous = make_action("click", unique=False)
        [node] = compile_action_cache(cache_of(ambiguous, done(make_action)))

        click = node.interaction_action.click_element
        assert click.command is None
        assert click.prompt_instructions == 'the textbox "City"'
        assert "max_tries" not in node.interaction_action.model_fields_set

    def test_fixed_actions_map_to_their_optexity_equivalents(self, make_action):
        nodes = compile_action_cache(
            cache_of(
                make_action("navigate", {"url": "https://example.com/a"}),
                make_action("search", {"query": "gdp india", "engine": "bing"}),
                make_action("go_back"),
                make_action("send_keys", {"keys": "Enter"}),
                done(make_action),
            )
        )

        go_to, search, back, key = (n.interaction_action for n in nodes)
        assert go_to.go_to_url.url == "https://example.com/a"
        assert search.go_to_url.url == "https://www.bing.com/search?q=gdp+india"
        assert back.go_back is not None
        assert key.key_press.type == "enter"

    def test_key_combinations_are_not_compiled(self, make_action):
        cache = cache_of(
            make_action("send_keys", {"keys": "Control+a"}), done(make_action)
        )
        with pytest.raises(UncompilableAction, match="key_press"):
            compile_action_cache(cache)

    def test_unsuccessful_run_is_not_compiled(self, make_action):
        cache = cache_of(
            make_action("input", {"text": "SF"}), done(make_action, success=False)
        )
        with pytest.raises(UncompilableAction, match="did not report success"):
            compile_action_cache(cache)


def _agentic_automation(*nodes) -> Automation:
    return Automation.model_validate(
        {
            "url": "https://example.com/form",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": list(nodes),
        }
    )


AGENTIC_NODE = {
    "type": "action_node",
    "interaction_action": {
        "agentic_task": {
            "task": "fill the city",
            "max_steps": 5,
            "backend": "browser_use",
        }
    },
}
KEY_NODE = {"type": "action_node", "interaction_action": {"key_press": {"type": "tab"}}}


def _write_cache(logs, step: int, cache: ActionCache) -> None:
    (logs / f"step_{step}").mkdir(parents=True)
    (logs / f"step_{step}" / "action_cache.json").write_text(cache.model_dump_json())


class TestBuildCachedAutomation:
    def test_replaces_only_the_cached_agentic_node(self, make_action, tmp_path):
        _write_cache(
            tmp_path,
            1,
            cache_of(
                make_action("input", {"text": "SF"}),
                make_action("click", {"index": 3}, started_download=True),
                done(make_action),
            ),
        )

        built = build_cached_automation(
            _agentic_automation(KEY_NODE, AGENTIC_NODE), tmp_path
        )

        key, typed, clicked = (n.interaction_action for n in built.nodes)
        assert key.key_press.type == "tab"
        assert typed.input_text.input_text == "SF"
        assert clicked.click_element.expect_download is True
        assert built.expected_downloads == 1

    def test_uncompilable_cache_keeps_the_agentic_task(self, make_action, tmp_path):
        _write_cache(
            tmp_path,
            0,
            cache_of(make_action("extract", {"query": "price"}), done(make_action)),
        )

        built = build_cached_automation(_agentic_automation(AGENTIC_NODE), tmp_path)

        [node] = built.nodes
        assert node.interaction_action.agentic_task.task == "fill the city"

    def test_agentic_node_without_a_cache_is_left_alone(self, tmp_path):
        built = build_cached_automation(_agentic_automation(AGENTIC_NODE), tmp_path)
        assert built.nodes[0].interaction_action.agentic_task is not None
