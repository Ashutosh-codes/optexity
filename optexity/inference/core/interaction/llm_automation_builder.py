"""LLM-built cached automation: docs + action cache in, validated automation out.

For each top-level ``agentic_task`` node with an action cache, the LLM gets the
relevant optexity docs, the cache, and the code-built draft
(``compile_action_cache``) as a reference, and returns the replacement nodes plus
any ``input_parameters`` it extracted from typed values. A response is accepted
only if it passes Pydantic validation and these checks:

- every ``command`` is a locator the cache verified unique on the live page;
- every typed or selected value, URL, and key is one the agent actually used;
- every ``{name[i]}`` placeholder is declared and every declared parameter used;
- the number of ``expect_download`` clicks equals the downloads the run started.

Otherwise the errors are sent back and the LLM retries, up to ``max_attempts``.
A node whose attempts all fail is kept as the original ``agentic_task``.
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from optexity.inference.core.interaction.cached_automation import (
    _REPLAYABLE_KEYS,
    _SEARCH_URLS,
    ACTION_CACHE_FILENAME,
    UncompilableAction,
    _describe,
    compile_action_cache,
)
from optexity.inference.models import get_llm_model
from optexity.inference.models.llm_model import parse_json_from_completion
from optexity.schema.action_cache import ActionCache
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.token_usage import TokenUsage
from optexity.utils.llm_settings import llm_settings

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[4] / "docs" / "docs"
DOC_PAGES = (
    "building-automations/automation-structure.mdx",
    "action-types/interaction-action.mdx",
    "building-automations/parameters.mdx",
    "advanced/locators.mdx",
    "advanced/downloads-files.mdx",
    "advanced/timing-retries.mdx",
)
_PLACEHOLDER = re.compile(r"\{(\w+)\[(\d+|index)\]\}")
_ALLOWED_INTERACTIONS = {
    "input_text",
    "click_element",
    "select_option",
    "key_press",
    "go_to_url",
    "go_back",
    "upload_file",
}

SYSTEM_PROMPT = """You convert a recorded browser-agent run into a deterministic Optexity automation.

You get the Optexity docs, the agent's goal, every action it took (with live-verified Playwright locators), and a draft compiled by code. Return the action nodes that replace the agentic task.

Rules:
- Replay the run faithfully: one node per action, in order, skipping only actions with category "redundant".
- Use only action_node with interaction_action of type input_text, click_element, select_option, key_press, go_to_url, go_back or upload_file.
- "command" must be copied exactly from the action's "locators" list (prefer the first). If the list is empty, omit "command" and rely on "prompt_instructions".
- Always write "prompt_instructions": a short description of the element from its visible label or name, for the LLM fallback.
- Set "expect_download": true on a click whose action has "started_download": true, and only there.
- When "command" is set, set interaction_action "max_tries" to 3. When the action has "same_page": true, set the node's "end_sleep_time" to 0.
- Values a user would change between runs (names, addresses, search terms) become input_parameters: declare each as a list holding the recorded value, e.g. {"city": ["SF"]}, and reference it as "{city[0]}" in input_text, select_values or command. Use descriptive snake_case names. Never parameterize values that are fixed UI labels.
- key_press "type" must be one of: enter, tab, space, /, 0-9.

Respond with JSON only, no prose:
{"input_parameters": {"<name>": ["<recorded value>"]}, "nodes": [<action_node>, ...]}"""


class BuildAttempt(BaseModel):
    node_index: int
    attempt: int
    errors: list[str] = Field(default_factory=list)


class LLMBuildReport(BaseModel):
    model: str
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    attempts: list[BuildAttempt] = Field(default_factory=list)
    kept_agentic: list[int] = Field(default_factory=list)


class _Response(BaseModel):
    input_parameters: dict[str, list[str]] = Field(default_factory=dict)
    nodes: list[dict]


def build_cached_automation_with_llm(
    automation: Automation,
    logs_directory: Path,
    model_name: str | None = None,
    max_attempts: int = 3,
    docs_dir: Path = DOCS_DIR,
) -> tuple[Automation, LLMBuildReport]:
    model_name = model_name or llm_settings.LLM_MODEL
    model = get_llm_model(model_name, use_structured_output=False)
    docs = _load_docs(docs_dir)
    report = LLMBuildReport(model=model_name)

    built = automation.model_copy(deep=True)
    params = dict(built.parameters.input_parameters)
    nodes = []
    for i, node in enumerate(built.nodes):
        cache_path = logs_directory / f"step_{i}" / ACTION_CACHE_FILENAME
        task = _agentic_task(node)
        if task is None or not cache_path.exists():
            nodes.append(node)
            continue
        cache = ActionCache.model_validate_json(cache_path.read_text())
        if not any(a.is_done and a.success for a in cache.actions):
            logger.warning(
                f"Keeping agentic_task node {i}: agent did not report success"
            )
            report.kept_agentic.append(i)
            nodes.append(node)
            continue

        prompt = _prompt(docs, automation.url, task, cache)
        response = None
        for attempt in range(1, max_attempts + 1):
            text, usage = model.get_model_response(
                prompt, system_instruction=SYSTEM_PROMPT
            )
            report.tokens += usage
            response, errors = _check(text, cache, params)
            report.attempts.append(
                BuildAttempt(node_index=i, attempt=attempt, errors=errors)
            )
            if not errors:
                break
            logger.info(f"Node {i} attempt {attempt} rejected: {errors}")
            prompt = _retry_prompt(prompt, text, errors)
            response = None

        if response is None:
            logger.warning(
                f"Keeping agentic_task node {i}: no valid build in {max_attempts} attempts"
            )
            report.kept_agentic.append(i)
            nodes.append(node)
            continue
        params.update(response.input_parameters)
        nodes.extend(ActionNode.model_validate(n) for n in response.nodes)

    built.nodes = nodes
    built.parameters.input_parameters = params
    built.expected_downloads = max(
        built.expected_downloads, sum(_expects_download(n) for n in nodes)
    )
    return Automation.model_validate(built.model_dump()), report


def _check(
    text: str, cache: ActionCache, existing_params: dict
) -> tuple[_Response | None, list[str]]:
    """Parse and validate one LLM answer; return it with every problem found."""
    try:
        response = parse_json_from_completion(text, _Response)
    except ValueError:
        return None, ["Response is not the requested JSON object."]

    errors = []
    nodes: list[ActionNode] = []
    for k, raw in enumerate(response.nodes):
        try:
            nodes.append(ActionNode.model_validate(raw))
        except ValidationError as e:
            errors.append(f"nodes[{k}] is not a valid action_node: {e}")
    if errors:
        return None, errors

    params = {**existing_params, **response.input_parameters}
    clashes = existing_params.keys() & response.input_parameters.keys()
    errors += [
        f"input parameter '{name}' already exists in the automation" for name in clashes
    ]
    errors += _check_against_cache(nodes, cache, params)

    used = {
        name for n in nodes for name, _ in _PLACEHOLDER.findall(n.model_dump_json())
    }
    errors += [
        f"placeholder '{{{name}[...]}}' is not declared in input_parameters"
        for name in used - params.keys()
    ]
    errors += [
        f"input parameter '{name}' is declared but never used"
        for name in response.input_parameters.keys() - used
    ]

    try:
        Automation.model_validate(
            {
                "url": "about:blank",
                "parameters": {"input_parameters": params, "generated_parameters": {}},
                "nodes": response.nodes,
            }
        )
    except ValidationError as e:
        errors.append(f"automation does not validate: {e}")
    return (None if errors else response), errors


def _check_against_cache(
    nodes: list[ActionNode], cache: ActionCache, params: dict
) -> list[str]:
    actions = [a for a in cache.actions if not a.is_done]
    locators = {loc.locator for a in actions for loc in a.locators if loc.is_unique}
    texts = {
        str(a.params.get("text"))
        for a in actions
        if a.action_name in ("input", "select_dropdown")
    }
    urls = {a.params["url"] for a in actions if a.action_name == "navigate"} | {
        _SEARCH_URLS.get(a.params.get("engine", "duckduckgo"), "").format(
            query=a.params.get("query", "")
        )
        for a in actions
        if a.action_name == "search"
    }
    files = {a.params.get("path") for a in actions if a.action_name == "upload_file"}
    downloads = sum(a.started_download for a in actions)

    errors = []
    for k, node in enumerate(nodes):
        interaction = node.interaction_action
        kinds = (
            [
                name
                for name in interaction.model_fields_set
                if getattr(interaction, name) is not None
                and isinstance(getattr(interaction, name), BaseModel)
            ]
            if interaction is not None
            else []
        )
        if not kinds or any(kind not in _ALLOWED_INTERACTIONS for kind in kinds):
            errors.append(
                f"nodes[{k}] must be one of {sorted(_ALLOWED_INTERACTIONS)}, got {kinds or 'none'}"
            )
            continue
        action = getattr(interaction, kinds[0])
        command = getattr(action, "command", None)
        if command and _render(command, params) not in locators:
            errors.append(
                f"nodes[{k}] command {command!r} is not one of the cache's verified locators"
            )
        if (
            kinds[0] == "input_text"
            and _render(str(action.input_text), params) not in texts
        ):
            errors.append(
                f"nodes[{k}] input_text {action.input_text!r} was never typed by the agent"
            )
        if kinds[0] == "select_option" and any(
            _render(str(v), params) not in texts for v in action.select_values or [None]
        ):
            errors.append(
                f"nodes[{k}] select_values {action.select_values!r} were never selected by the agent"
            )
        if kinds[0] == "go_to_url" and _render(action.url, params) not in urls:
            errors.append(
                f"nodes[{k}] url {action.url!r} was never opened by the agent"
            )
        if (
            kinds[0] == "upload_file"
            and _render(str(action.file_path), params) not in files
        ):
            errors.append(
                f"nodes[{k}] file_path {action.file_path!r} was never uploaded by the agent"
            )
        if kinds[0] == "key_press" and (
            not isinstance(action.type, str) or action.type not in _REPLAYABLE_KEYS
        ):
            errors.append(
                f"nodes[{k}] key_press type {action.type!r} is not replayable"
            )

    expected = sum(_expects_download(n) for n in nodes)
    if expected != downloads:
        errors.append(
            f"{expected} click(s) have expect_download, but the run started {downloads} download(s)"
        )
    return errors


def _prompt(docs: str, url: str, task: str, cache: ActionCache) -> str:
    try:
        draft = [
            n.model_dump(exclude_defaults=True) for n in compile_action_cache(cache)
        ]
    except UncompilableAction as e:
        draft = f"code could not compile this run: {e}"
    actions = [
        {
            "action": a.action_name,
            "params": a.params,
            "category": a.category,
            "reason": a.reason,
            "element": _describe(a.element) if a.element else None,
            "locators": [loc.locator for loc in a.locators if loc.is_unique],
            "url_before": a.url_before,
            "url_after": a.url_after,
            "same_page": a.url_after is not None and a.url_after == a.url_before,
            "started_download": a.started_download,
        }
        for a in cache.actions
        if not a.is_done
    ]
    return (
        f"[OPTEXITY DOCS]\n{docs}\n[/OPTEXITY DOCS]\n\n"
        f"Start URL: {url}\nAgent goal: {task}\n\n"
        f"[RECORDED ACTIONS]\n{json.dumps(actions, indent=1)}\n[/RECORDED ACTIONS]\n\n"
        f"[CODE DRAFT]\n{json.dumps(draft, indent=1)}\n[/CODE DRAFT]"
    )


def _retry_prompt(prompt: str, previous: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return (
        f"{prompt}\n\n[YOUR PREVIOUS ANSWER]\n{previous}\n[/YOUR PREVIOUS ANSWER]\n\n"
        f"It was rejected:\n{listed}\nReturn a corrected answer as JSON only."
    )


def _load_docs(docs_dir: Path) -> str:
    missing = [p for p in DOC_PAGES if not (docs_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Optexity docs not found under {docs_dir}: {missing}")
    return "\n\n".join(f"--- {p} ---\n{(docs_dir / p).read_text()}" for p in DOC_PAGES)


def _render(value: str, params: dict) -> str:
    """``value`` with each ``{name[i]}`` replaced by its declared value."""

    def value_of(match: re.Match) -> str:
        values = params.get(match.group(1), [])
        i = int(match.group(2)) if match.group(2).isdigit() else 0
        return str(values[i]) if i < len(values) else match.group(0)

    return _PLACEHOLDER.sub(value_of, value)


def _agentic_task(node) -> str | None:
    interaction = getattr(node, "interaction_action", None)
    agentic = interaction.agentic_task if interaction is not None else None
    return agentic.task if agentic is not None else None


def _expects_download(node: ActionNode) -> bool:
    interaction = node.interaction_action
    click = interaction.click_element if interaction is not None else None
    return bool(click and click.expect_download)
