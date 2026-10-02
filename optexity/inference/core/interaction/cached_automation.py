import logging
from pathlib import Path
from urllib.parse import quote_plus

from optexity.schema.action_cache import ActionCache, CachedAction, CachedElement
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    KeyPressAction,
    SelectOptionAction,
    UploadFileAction,
)
from optexity.schema.automation import ActionNode, Automation

logger = logging.getLogger(__name__)

ACTION_CACHE_FILENAME = "action_cache.json"
# Each try waits up to 1 s (``max_timeout_seconds_per_try``), so this still gives
# late-rendering elements a few seconds after a navigation.
CACHED_LOCATOR_MAX_TRIES = 3

# Same URLs browser-use's ``search`` action opens.
_SEARCH_URLS = {
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "google": "https://www.google.com/search?q={query}&udm=14",
    "bing": "https://www.bing.com/search?q={query}",
}

# ``handle_key_press`` silently ignores any key it has no branch for, including
# combinations, so only these single keys are safe to compile.
_REPLAYABLE_KEYS = {"enter", "tab", "space", "/", *"0123456789"}


class UncompilableAction(Exception):
    """The cache holds an action no deterministic node can reproduce."""


def build_cached_automation(automation: Automation, logs_directory: Path) -> Automation:
    """Return a copy of ``automation`` with each cached ``agentic_task`` compiled.

    A top-level ``agentic_task`` node at position ``i`` is replaced by the nodes
    compiled from ``logs_directory/step_<i>/action_cache.json``. ``step_<i>``
    matches node ``i`` only for top-level action nodes, which run in order, so
    nested nodes are left untouched. A node with no cache, or whose cache holds an
    action that cannot be compiled, is kept as the original ``agentic_task``.
    """
    cached = automation.model_copy(deep=True)
    nodes = []
    for i, node in enumerate(cached.nodes):
        cache_path = logs_directory / f"step_{i}" / ACTION_CACHE_FILENAME
        is_agentic = (
            isinstance(node, ActionNode)
            and node.interaction_action is not None
            and node.interaction_action.agentic_task is not None
        )
        if not is_agentic or not cache_path.exists():
            nodes.append(node)
            continue

        cache = ActionCache.model_validate_json(cache_path.read_text())
        try:
            compiled = compile_action_cache(cache)
        except UncompilableAction as e:
            logger.warning(f"Keeping agentic_task node {i}: {e}")
            nodes.append(node)
            continue
        logger.info(
            f"Node {i}: {len(cache.actions)} cached action(s) -> {len(compiled)} node(s)"
        )
        nodes.extend(compiled)

    cached.nodes = nodes
    cached.expected_downloads = max(
        cached.expected_downloads, sum(_expects_download(node) for node in nodes)
    )
    return Automation.model_validate(cached.model_dump())


def _expects_download(node) -> bool:
    interaction = getattr(node, "interaction_action", None)
    click = interaction.click_element if interaction else None
    return bool(click and click.expect_download)


def compile_action_cache(cache: ActionCache) -> list[ActionNode]:
    """Deterministic nodes for one ``agentic_task`` run, ``redundant`` actions dropped.

    Element actions use the cache's best (live-verified unique) locator as
    ``command``, with ``prompt_instructions`` as the fallback. Element actions
    with no unique locator get no ``command``, so optexity resolves them with one
    LLM call on the axtree instead of a full agent loop.
    """
    if not any(action.is_done and action.success for action in cache.actions):
        raise UncompilableAction("agent did not report success")
    if not all(action.category for action in cache.actions):
        raise UncompilableAction("cache is not classified")
    nodes = [
        _node(action) for action in cache.actions if action.category != "redundant"
    ]
    if not nodes:
        raise UncompilableAction("no actions left after dropping redundant ones")
    return nodes


def _node(action: CachedAction) -> ActionNode:
    """Wrap the action's interaction, tuned for a replay of a known-good run.

    A failing live-verified locator rarely recovers by retrying, so it gets
    ``CACHED_LOCATOR_MAX_TRIES`` before the ``prompt_instructions`` fallback
    instead of optexity's default 10 (about 11 s). An action that left the URL
    unchanged skips the post-action ``load`` wait.
    """
    interaction = _interaction(action)
    if action.best_locator is not None:
        interaction.max_tries = CACHED_LOCATOR_MAX_TRIES
    same_page = action.url_after is not None and action.url_after == action.url_before
    return ActionNode(
        type="action_node",
        interaction_action=interaction,
        **({"end_sleep_time": 0.0} if same_page else {}),
    )


def _interaction(action: CachedAction) -> InteractionAction:
    name, params = action.action_name, action.params

    if name in {"input", "click", "select_dropdown", "upload_file"}:
        if action.element is None:
            raise UncompilableAction(f"'{name}' has no resolved element")
        best = action.best_locator
        target = {
            "command": best.locator if best else None,
            "prompt_instructions": _describe(action.element),
        }
        if name == "input":
            return InteractionAction(
                input_text=InputTextAction(input_text=params["text"], **target)
            )
        if name == "click":
            return InteractionAction(
                click_element=ClickElementAction(
                    expect_download=action.started_download, **target
                )
            )
        if name == "select_dropdown":
            return InteractionAction(
                select_option=SelectOptionAction(
                    select_values=[params["text"]], **target
                )
            )
        return InteractionAction(
            upload_file=UploadFileAction(file_path=params["path"], **target)
        )

    if name == "navigate":
        return InteractionAction(
            go_to_url=GoToUrlAction(
                url=params["url"], new_tab=params.get("new_tab", False)
            )
        )
    if name == "search":
        template = _SEARCH_URLS.get(params.get("engine", "duckduckgo"))
        if template is None:
            raise UncompilableAction(f"unknown search engine {params.get('engine')!r}")
        url = template.format(query=quote_plus(params["query"]))
        return InteractionAction(go_to_url=GoToUrlAction(url=url))
    if name == "go_back":
        return InteractionAction(go_back=GoBackAction())
    if name == "send_keys":
        key = params["keys"].lower()
        if key not in _REPLAYABLE_KEYS:
            raise UncompilableAction(f"key_press cannot replay {params['keys']!r}")
        return InteractionAction(key_press=KeyPressAction(type=key))

    raise UncompilableAction(f"no deterministic node for '{name}' ({action.reason})")


def _describe(element: CachedElement) -> str:
    """Element description for optexity's LLM fallback, from the strongest signals
    the cache has (accessible name, then identifying attributes, then text)."""
    kind = element.role or element.tag
    attrs = element.attributes
    if element.name:
        return f'the {kind} "{element.name}"'
    for attr in ("placeholder", "aria-label", "title", "name", "id"):
        if attrs.get(attr):
            return f'the {kind} with {attr}="{attrs[attr]}"'
    if element.text:
        return f'the {kind} with text "{element.text}"'
    return f"the {kind} at xpath {element.xpath}"
