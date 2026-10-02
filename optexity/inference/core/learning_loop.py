"""Run → cache → rebuild loop over the local inference server.

Each round runs the current automation through ``POST /inference`` (the server's
local override reads it from ``live_path``), reads that run's logs, and derives
the next automation:

1. **Compile**: ``agentic_task`` nodes become deterministic nodes from the round's
   action cache (``build_cached_automation``, or the LLM builder with
   ``builder="llm"``; its tokens are reported in the round's changes).
2. **Heal**: a node whose ``command`` failed and was recovered by the
   ``prompt_instructions`` fallback takes the locator the fallback used.
3. **Prune**: once a round is clean (no compile or heal changes), one candidate
   segment is removed and the next round must reproduce the same final URL and
   downloads with no locator failure; otherwise it is restored and not tried
   again. Candidates are same-page actions on a page the run later leaves, never
   a download or anything on the final page, since URL and downloads are the only
   outcome signals and would not notice a missing form field there. A rejected
   segment is retried as shorter prefixes (the actions right before the
   navigation are the likeliest to be needed). Trial rounds run with
   ``skip_prompt`` so a bad prune fails in seconds instead of paying for the
   LLM and agentic fallbacks.

The loop stops when a round changes nothing or after ``max_rounds``.
"""

import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from optexity.inference.core.interaction.cached_automation import (
    build_cached_automation,
)
from optexity.inference.core.interaction.llm_automation_builder import (
    build_cached_automation_with_llm,
)
from optexity.schema.automation import ActionNode, Automation

logger = logging.getLogger(__name__)

NodeKind = Literal["command", "fallback", "agentic", "other"]

_STATUS_LINE = re.compile(r"Task (\S+) completed with status (\w+)")
_LOCATOR_FAILURE = re.compile(r"Action failed after \d+ tries")
_ACTION_METHODS = (
    "fill",
    "type",
    "click",
    "dblclick",
    "select_option",
    "check",
    "uncheck",
    "hover",
    "set_input_files",
)


class NodeOutcome(BaseModel):
    index: int
    kind: NodeKind
    url_before: str | None = None


class RoundResult(BaseModel):
    round: int
    task_id: str
    status: str
    tokens: int
    nodes_done_s: float | None
    task_s: float | None
    final_url: str | None
    downloads: int
    locator_failures: int
    outcomes: list[NodeOutcome]
    changes: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Succeeded with every node replayed from its locator."""
        return (
            self.status == "success"
            and self.locator_failures == 0
            and all(o.kind in ("command", "other") for o in self.outcomes)
        )


class _PendingPrune(BaseModel):
    signature: str
    reference: RoundResult
    before: Automation
    candidate: Automation


def run_learning_loop(
    automation: Automation,
    endpoint_name: str,
    out_dir: Path,
    live_path: Path,
    runs_dir: Path = Path("runs"),
    server_url: str = "http://localhost:9000",
    max_rounds: int = 5,
    builder: Literal["code", "llm"] = "code",
) -> tuple[Automation, list[RoundResult]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RoundResult] = []
    rejected: set[str] = set()
    pending: _PendingPrune | None = None

    for round_no in range(1, max_rounds + 1):
        round_path = out_dir / f"round_{round_no}.json"
        round_path.write_text(_dump(automation))
        live_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(round_path, live_path)

        task_id = _submit(server_url, endpoint_name)
        logs = runs_dir / task_id / "logs"
        status = _wait_for_status(logs / "optexity.log", task_id)
        _check_ran(logs, automation)
        result = _read_round(round_no, task_id, status, logs, runs_dir / task_id)

        if pending is not None:
            # A rejected prune restores the previous automation and its result,
            # so the next candidate is chosen without rerunning it.
            kept = _reproduces(result, pending.reference)
            result.changes.append(
                f"prune {'kept' if kept else 'reverted'}: {pending.signature}"
            )
            if not kept:
                rejected.add(pending.signature)
            base, base_result = (
                (pending.candidate, result)
                if kept
                else (pending.before, pending.reference)
            )
            pending = None
            next_automation, changes = base, []
        else:
            base, base_result = automation, result
            next_automation, changes = _refine(automation, result, logs, builder)

        if not changes and base_result.clean:
            segment = _prune_candidate(base, base_result, rejected)
            if segment is not None:
                signature = _signature(base, segment)
                candidate = _without(base, segment)
                pending = _PendingPrune(
                    signature=signature,
                    reference=base_result,
                    before=base,
                    candidate=candidate,
                )
                next_automation = _fail_fast(candidate)
                changes = [f"trying prune: {signature}"]

        result.changes.extend(changes)
        results.append(result)
        _report(result)
        automation = next_automation
        if not changes:
            break

    if pending is not None:
        automation = pending.before
    (out_dir / "final.json").write_text(_dump(automation))
    (out_dir / "rounds.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2)
    )
    return automation, results


def _refine(
    automation: Automation,
    result: RoundResult,
    logs: Path,
    builder: Literal["code", "llm"] = "code",
) -> tuple[Automation, list[str]]:
    """Heal fallen-back locators, then compile agentic nodes. Healing goes first
    because both address nodes by the index they ran at."""
    healed = automation.model_copy(deep=True)
    changes = []
    for outcome in result.outcomes:
        if outcome.kind != "fallback":
            continue
        command = _fallback_command(logs / f"step_{outcome.index}")
        target = _locator_action(healed.nodes[outcome.index])
        if command and target is not None and target.command != command:
            changes.append(f"healed node {outcome.index}: {command}")
            target.command = command

    note = ""
    if builder == "llm" and any(o.kind == "agentic" for o in result.outcomes):
        compiled, report = build_cached_automation_with_llm(healed, logs)
        note = (
            f" (LLM: {len(report.attempts)} attempt(s), {report.tokens.total_tokens} tokens"
            f"{f', kept agentic {report.kept_agentic}' if report.kept_agentic else ''})"
        )
    else:
        compiled = build_cached_automation(healed, logs)
    if _dump(compiled) != _dump(healed):
        changes.append(
            f"compiled agentic nodes: {len(healed.nodes)} -> {len(compiled.nodes)} nodes{note}"
        )
    return compiled, changes


def _prune_candidate(
    automation: Automation, result: RoundResult, rejected: set[str]
) -> list[int] | None:
    """First untried prefix, longest first, of a run of consecutive same-page
    nodes on a page the run leaves.

    Outcome ``i`` is node ``i``: logs are per top-level node, and only clean
    rounds (every node ran) get here.
    """
    outcomes = result.outcomes
    url_after = [o.url_before for o in outcomes[1:]] + [result.final_url]

    def is_candidate(i: int) -> bool:
        page = outcomes[i].url_before
        if page is None or url_after[i] != page or page == result.final_url:
            return False
        if _expects_download(automation.nodes[i]):
            return False
        return any(after != page for after in url_after[i:])

    segments: list[list[int]] = []
    for i in range(len(outcomes)):
        if not is_candidate(i):
            continue
        last = segments[-1] if segments else None
        if (
            last
            and last[-1] == i - 1
            and outcomes[last[0]].url_before == outcomes[i].url_before
        ):
            last.append(i)
        else:
            segments.append([i])
    prefixes = (s[:k] for s in segments for k in range(len(s), 0, -1))
    return next(
        (p for p in prefixes if _signature(automation, p) not in rejected), None
    )


def _reproduces(result: RoundResult, reference: RoundResult) -> bool:
    return (
        result.clean
        and result.final_url == reference.final_url
        and result.downloads == reference.downloads
    )


def _read_round(
    round_no: int, task_id: str, status: str, logs: Path, task_dir: Path
) -> RoundResult:
    steps = sorted(
        (p for p in logs.glob("step_*") if (p / "state.json").exists()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    states = [json.loads((p / "state.json").read_text()) for p in steps]
    node_steps = [p for p in steps if (p / "action_node.json").exists()]
    outcomes = [
        NodeOutcome(
            index=int(p.name.split("_")[1]),
            kind=_node_kind(p),
            url_before=json.loads((p / "state.json").read_text()).get("url"),
        )
        for p in node_steps
    ]

    started = _time(states[0].get("started_at")) if states else None
    last_node = (
        json.loads((node_steps[-1] / "state.json").read_text()) if node_steps else {}
    )
    final = states[-1] if states else {}
    downloads_dir = task_dir / "downloads"
    return RoundResult(
        round=round_no,
        task_id=task_id,
        status=status,
        tokens=(final.get("token_usage") or {}).get("total_tokens", 0),
        nodes_done_s=_elapsed(started, last_node.get("completed_at")),
        task_s=_elapsed(started, final.get("completed_at")),
        final_url=final.get("url"),
        downloads=(
            sum(1 for f in downloads_dir.iterdir() if f.is_file())
            if downloads_dir.exists()
            else 0
        ),
        locator_failures=len(
            _LOCATOR_FAILURE.findall(
                (logs / "optexity.log").read_text(errors="replace")
            )
        ),
        outcomes=outcomes,
    )


def _node_kind(step: Path) -> NodeKind:
    node = json.loads((step / "action_node.json").read_text())
    interaction = node.get("interaction_action") or {}
    if "agentic_task" in interaction:
        return "agentic"
    if (step / "llm_response.json").exists():
        return "fallback"
    if any(k in interaction for k in ("input_text", "click_element", "select_option")):
        return "command"
    return "other"


def _fallback_command(step: Path) -> str | None:
    """The locator optexity's prompt fallback acted on, as a ``command``. Recorded
    as ``page.<locator>.<method>(...)``; the method suffix is stripped."""
    path = step / "locator_candidates.json"
    if not path.exists():
        return None
    candidates = json.loads(path.read_text())
    if not candidates:
        return None
    locator = candidates[0]["locator"].removeprefix("page.")
    cut = max(locator.rfind(f".{method}(") for method in _ACTION_METHODS)
    return locator[:cut] if cut > 0 else None


def _locator_action(node):
    interaction = getattr(node, "interaction_action", None)
    if interaction is None:
        return None
    return (
        interaction.input_text
        or interaction.click_element
        or interaction.select_option
        or interaction.check
        or interaction.uncheck
        or interaction.hover
    )


def _expects_download(node) -> bool:
    action = _locator_action(node)
    return bool(getattr(action, "expect_download", False))


def _signature(automation: Automation, segment: list[int]) -> str:
    """Index-independent name for a segment, so a rejected prune stays rejected
    after earlier nodes are removed."""
    parts = []
    for i in segment:
        action = _locator_action(automation.nodes[i])
        node = automation.nodes[i]
        if action is not None and action.command:
            parts.append(action.command)
        elif isinstance(node, ActionNode) and node.interaction_action is not None:
            parts.append(
                json.dumps(node.interaction_action.model_dump(exclude_defaults=True))
            )
    return " + ".join(parts)


def _without(automation: Automation, segment: list[int]) -> Automation:
    pruned = automation.model_copy(deep=True)
    pruned.nodes = [n for i, n in enumerate(pruned.nodes) if i not in segment]
    return Automation.model_validate(pruned.model_dump())


def _fail_fast(automation: Automation) -> Automation:
    """Trial copy whose locator nodes skip the prompt fallback."""
    trial = automation.model_copy(deep=True)
    for node in trial.nodes:
        action = _locator_action(node)
        if action is not None and action.command:
            action.skip_prompt = True
    return Automation.model_validate(trial.model_dump())


def _submit(server_url: str, endpoint_name: str) -> str:
    response = httpx.post(
        f"{server_url}/inference",
        json={"endpoint_name": endpoint_name, "input_parameters": {}},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["task_id"]


def _wait_for_status(log_path: Path, task_id: str, timeout_s: float = 900) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists():
            for match in _STATUS_LINE.finditer(log_path.read_text(errors="replace")):
                if match.group(1) == task_id:
                    return match.group(2)
        time.sleep(2)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout_s:.0f}s")


def _check_ran(logs: Path, automation: Automation) -> None:
    """Fail fast if the server is not reading ``live_path``."""
    first = logs / "step_0" / "action_node.json"
    if not automation.nodes or not first.exists():
        return
    ran = json.loads(first.read_text()).get("interaction_action")
    expected = automation.nodes[0].model_dump().get("interaction_action")
    if ran is not None and expected is not None:
        ran_kind = next(iter(k for k, v in ran.items() if isinstance(v, dict)), None)
        expected_kind = next(
            iter(k for k, v in expected.items() if isinstance(v, dict)), None
        )
        if ran_kind != expected_kind:
            raise RuntimeError(
                "The server ran a different automation; point its "
                "OPTEXITY_LOCAL_AUTOMATION at the loop's live path."
            )


def _report(result: RoundResult) -> None:
    kinds = [o.kind for o in result.outcomes]
    logger.info(
        f"Round {result.round} ({result.task_id}): {result.status}, "
        f"tokens {result.tokens}, steps {result.nodes_done_s}s, task {result.task_s}s, "
        f"downloads {result.downloads}, nodes {kinds}; changes: {result.changes or 'none'}"
    )


def _dump(automation: Automation) -> str:
    return json.dumps(automation.model_dump(exclude_defaults=True), indent=2) + "\n"


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _elapsed(start: datetime | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    return round((datetime.fromisoformat(end) - start).total_seconds(), 1)
