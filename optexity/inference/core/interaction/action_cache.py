import logging
import os
import time
from pathlib import Path

from browser_use.agent.views import ActionEvent
from browser_use.dom.views import EnhancedDOMTreeNode

from optexity.inference.core.interaction.utils import LocatorExtraction, Page
from optexity.inference.infra.browser import Browser
from optexity.schema.action_cache import (
    ActionCache,
    ActionCategory,
    CachedAction,
    CachedElement,
    CachedLocator,
)

logger = logging.getLogger(__name__)

# Actions that only inform the agent's next decision and leave nothing on the page a
# replay depends on (Playwright scrolls targets into view and waits on its own).
_EXPLORATORY_ACTIONS = {
    "wait",
    "scroll",
    "find_text",
    "screenshot",
    "dropdown_options",
    "write_file",
    "read_file",
    "replace_file",
}
_ELEMENT_ACTIONS = {"click", "input", "select_dropdown", "upload_file"}
# Element actions whose effect is fully replaced by a later one on the same element.
_OVERWRITING_ACTIONS = {"input", "select_dropdown", "upload_file"}
_FIXED_ACTIONS = {"navigate", "search", "go_back", "send_keys"}


def _element_key(action: CachedAction) -> tuple[str, str | None, str]:
    xpath = action.element.xpath if action.element else None
    return (action.action_name, action.url_before, xpath or "")


def classify_actions(cache: ActionCache) -> None:
    """Set ``category`` and ``reason`` on every cached action.

    - ``deterministic``: replayable without an LLM (element action with a unique
      locator, or a fixed navigation/key press).
    - ``redundant``: safe to drop (failed, exploratory, overwritten later, the
      ``done`` report, or navigating to the page already open).
    - ``needs_llm``: depends on page content or has no unique locator (``extract``,
      LLM-written JS, tab ids that change per run).
    """
    last_write_by_element = {
        _element_key(action): i
        for i, action in enumerate(cache.actions)
        if action.action_name in _OVERWRITING_ACTIONS
        and action.element
        and not action.error
    }

    for i, action in enumerate(cache.actions):
        later = last_write_by_element.get(_element_key(action))
        overwritten_by = later if later is not None and later != i else None
        action.category, action.reason = _classify(action, overwritten_by)


def _classify(
    action: CachedAction, overwritten_by: int | None
) -> tuple[ActionCategory, str]:
    name = action.action_name
    if action.error:
        return "redundant", "action failed"
    if name == "done":
        return "redundant", "completion report, no browser effect"
    if name in _EXPLORATORY_ACTIONS:
        return "redundant", "exploratory, no lasting page effect"
    if name in _ELEMENT_ACTIONS:
        if overwritten_by is not None:
            return "redundant", f"overwritten by action {overwritten_by}"
        if action.best_locator is not None:
            return "deterministic", f"unique {action.best_locator.kind} locator"
        return "needs_llm", "no unique locator; replay via prompt_instructions"
    if name == "navigate" and action.params.get("url") == action.url_before:
        return "redundant", "already on that page"
    if name in _FIXED_ACTIONS:
        return "deterministic", "fixed parameters, no element"
    if name == "extract":
        return "needs_llm", "LLM reads page content"
    if name in {"switch", "close"}:
        return "needs_llm", "tab ids change between runs"
    return "needs_llm", f"no deterministic mapping for '{name}'"


class ActionCacheRecorder:
    """Builds an ``ActionCache`` from browser-use's per-action callbacks.

    ``on_action_start`` runs while the targeted element is still on the live page,
    so every ``LocatorExtraction`` candidate is checked there with a Playwright
    ``count()``; after the action the page may have navigated away. Pass
    ``on_action_start``/``on_action_end`` as the Agent's
    ``register_action_start_callback``/``register_action_end_callback``.
    """

    def __init__(self, browser: Browser, task: str):
        self.browser = browser
        self.cache = ActionCache(task=task)
        self._pending: dict[tuple[int, int], tuple[CachedAction, float]] = {}
        self._pages_by_target: dict[str, Page] = {}
        self._downloads_seen = self._download_count()

    async def on_action_start(self, event: ActionEvent) -> None:
        self._attribute_downloads()
        action_data = event.action.model_dump(exclude_unset=True)
        action_name = next(iter(action_data), "unknown")
        page = await self._page_for_target(event.target_id)

        cached = CachedAction(
            step=event.step,
            action_index=event.action_index,
            action_name=action_name,
            params=action_data.get(action_name) or {},
            url_before=page.url if page else None,
        )
        if event.element is not None:
            cached.element = self._element_signals(event.element)
            cached.locators = await self._verified_locators(event.element, page)

        self._pending[(event.step, event.action_index)] = (cached, time.perf_counter())

    async def on_action_end(self, event: ActionEvent) -> None:
        pending = self._pending.pop((event.step, event.action_index), None)
        if pending is None:
            return
        cached, started = pending
        cached.duration_s = round(time.perf_counter() - started, 3)

        result = event.result
        if result is not None:
            cached.error = result.error
            cached.is_done = bool(result.is_done)
            cached.success = result.success
            cached.extracted_content = result.extracted_content

        page = await self._page_for_target(event.target_id)
        cached.url_after = page.url if page else None
        self.cache.actions.append(cached)

    def save(self, path: Path) -> None:
        self._attribute_downloads()
        classify_actions(self.cache)
        path.write_text(self.cache.model_dump_json(indent=2))
        logger.info(f"Action cache: {len(self.cache.actions)} action(s) -> {path}")

    def _download_count(self) -> int:
        """Downloads seen so far, over the channels optexity's ``handle_download``
        watches: files (incl. partial) in the temp download dir, captured file
        responses, and Playwright download events."""
        memory = self.browser.memory
        try:
            files = len(os.listdir(self.browser.temp_downloads_dir))
        except OSError:
            files = 0
        return files + len(memory.urls_to_downloads) + len(memory.raw_downloads)

    def _attribute_downloads(self) -> None:
        """Mark the last finished action as having started any download that
        appeared since the previous check. A download often lands after its click
        returns, so this runs when the next action starts and on save."""
        count = self._download_count()
        if count > self._downloads_seen and self.cache.actions:
            self.cache.actions[-1].started_download = True
        self._downloads_seen = count

    @staticmethod
    def _element_signals(element: EnhancedDOMTreeNode) -> CachedElement:
        ax = element.ax_node
        return CachedElement(
            tag=element.tag_name,
            attributes=element.attributes or {},
            role=ax.role if ax else None,
            name=ax.name if ax else None,
            text=element.get_meaningful_text_for_llm()[:200] or None,
            xpath=element.xpath,
            frame_id=element.frame_id,
        )

    @staticmethod
    async def _verified_locators(
        element: EnhancedDOMTreeNode, page: Page | None
    ) -> list[CachedLocator]:
        """All candidates best-first, each with its live match count. Uses the same
        ``eval(f"page.{command}")`` path the replay will use."""
        locators = []
        for score, kind, command in LocatorExtraction._scored_candidates(element):
            match_count = None
            if page is not None:
                try:
                    match_count = await eval(f"page.{command}", {"page": page}).count()
                except Exception as e:
                    logger.debug(f"Could not count {command}: {type(e).__name__}: {e}")
            locators.append(
                CachedLocator(
                    locator=command, kind=kind, score=score, match_count=match_count
                )
            )
        return locators

    async def _page_for_target(self, target_id: str | None) -> Page | None:
        """The Playwright page for browser-use's focused CDP target. Optexity's
        ``get_current_page`` returns the last-opened tab, which need not be the one
        browser-use is acting on."""
        if target_id is None or self.browser.context is None:
            return None
        page = self._pages_by_target.get(target_id)
        if page is not None and not page.is_closed():
            return page

        for candidate in self.browser.context.pages:
            if candidate in self._pages_by_target.values():
                continue
            try:
                session = await self.browser.context.new_cdp_session(candidate)
                info = await session.send("Target.getTargetInfo")
                await session.detach()
            except Exception as e:
                logger.debug(f"Could not read target id: {type(e).__name__}: {e}")
                continue
            self._pages_by_target[info["targetInfo"]["targetId"]] = candidate

        return self._pages_by_target.get(target_id)
