import json

from optexity.inference.core.learning_loop import (
    NodeOutcome,
    RoundResult,
    _fail_fast,
    _fallback_command,
    _prune_candidate,
    _reproduces,
    _signature,
)
from optexity.schema.automation import Automation

HOME = "https://example.com/"
COUNTRY = "https://example.com/country/IN"
INDICATOR = "https://example.com/indicator/gdp"


def _click(name: str, expect_download: bool = False) -> dict:
    click = {"command": f'get_by_role("link", name="{name}")'}
    if expect_download:
        click["expect_download"] = True
    return {"type": "action_node", "interaction_action": {"click_element": click}}


def _type(name: str, text: str) -> dict:
    return {
        "type": "action_node",
        "interaction_action": {
            "input_text": {
                "command": f'get_by_role("textbox", name="{name}")',
                "input_text": text,
            }
        },
    }


ENTER = {"type": "action_node", "interaction_action": {"key_press": {"type": "enter"}}}


def _automation(*nodes) -> Automation:
    return Automation.model_validate(
        {
            "url": HOME,
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": list(nodes),
        }
    )


def _round(
    pages: list[str],
    final_url: str,
    downloads: int = 1,
    status: str = "success",
    locator_failures: int = 0,
    kinds: list[str] | None = None,
) -> RoundResult:
    kinds = kinds or ["command"] * len(pages)
    return RoundResult(
        round=1,
        task_id="t",
        status=status,
        tokens=0,
        nodes_done_s=10.0,
        task_s=20.0,
        final_url=final_url,
        downloads=downloads,
        locator_failures=locator_failures,
        outcomes=[
            NodeOutcome(index=i, kind=kind, url_before=page)
            for i, (page, kind) in enumerate(zip(pages, kinds))
        ],
    )


# A dead-end search on the home page (nodes 0-2), then the path that worked.
FLOW = _automation(
    _type("Search", "India"),
    ENTER,
    _click("Search button"),
    _click("India"),
    _click("GDP"),
    _click("CSV", expect_download=True),
)
FLOW_PAGES = [HOME, HOME, HOME, HOME, COUNTRY, INDICATOR]


class TestPruneCandidate:
    def test_offers_the_same_page_run_on_a_page_the_flow_leaves(self):
        result = _round(FLOW_PAGES, final_url=INDICATOR)
        assert _prune_candidate(FLOW, result, rejected=set()) == [0, 1, 2]

    def test_retries_a_rejected_segment_as_shorter_prefixes(self):
        result = _round(FLOW_PAGES, final_url=INDICATOR)
        rejected = {_signature(FLOW, [0, 1, 2])}
        assert _prune_candidate(FLOW, result, rejected) == [0, 1]
        rejected.add(_signature(FLOW, [0, 1]))
        assert _prune_candidate(FLOW, result, rejected) == [0]

    def test_never_offers_actions_on_the_final_page(self):
        form = _automation(_type("City", "SF"), _type("Zip", "94105"))
        result = _round([HOME, HOME], final_url=HOME, downloads=0)
        assert _prune_candidate(form, result, rejected=set()) is None

    def test_never_offers_a_download_click(self):
        flow = _automation(_click("Report", expect_download=True), _click("Next"))
        result = _round([HOME, HOME], final_url=COUNTRY)
        assert _prune_candidate(flow, result, rejected=set()) is None


class TestReproduces:
    def test_same_outcome_without_failures_is_kept(self):
        reference = _round(FLOW_PAGES, final_url=INDICATOR)
        trial = _round(FLOW_PAGES[3:], final_url=INDICATOR)
        assert _reproduces(trial, reference)

    def test_locator_failures_reject_a_trial_that_reports_success(self):
        reference = _round(FLOW_PAGES, final_url=INDICATOR)
        trial = _round(FLOW_PAGES[3:], final_url=INDICATOR, locator_failures=3)
        assert not _reproduces(trial, reference)

    def test_different_final_page_or_downloads_reject_the_trial(self):
        reference = _round(FLOW_PAGES, final_url=INDICATOR)
        assert not _reproduces(_round(FLOW_PAGES, final_url=HOME), reference)
        assert not _reproduces(
            _round(FLOW_PAGES, final_url=INDICATOR, downloads=0), reference
        )

    def test_a_fallback_round_is_not_clean(self):
        result = _round([HOME], final_url=HOME, kinds=["fallback"])
        assert not result.clean


class TestHealingAndTrials:
    def test_fallback_command_strips_the_action_method(self, tmp_path):
        (tmp_path / "locator_candidates.json").write_text(
            json.dumps(
                [
                    {
                        "locator": "page.locator(\"xpath=html/body/input[2]\").fill('xyz')",
                        "kind": "xpath",
                        "score": 10,
                    }
                ]
            )
        )
        assert _fallback_command(tmp_path) == 'locator("xpath=html/body/input[2]")'

    def test_fallback_command_is_none_without_candidates(self, tmp_path):
        assert _fallback_command(tmp_path) is None

    def test_fail_fast_skips_the_prompt_only_on_locator_nodes(self):
        trial = _fail_fast(FLOW)
        typed, enter = trial.nodes[0].interaction_action, trial.nodes[1]
        assert typed.input_text.skip_prompt is True
        assert enter.interaction_action.key_press is not None
        assert FLOW.nodes[0].interaction_action.input_text.skip_prompt is False
