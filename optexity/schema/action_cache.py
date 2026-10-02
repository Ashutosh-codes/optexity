from typing import Any, Literal

from pydantic import BaseModel, Field

ActionCategory = Literal["deterministic", "redundant", "needs_llm"]


class CachedLocator(BaseModel):
    """A Playwright locator candidate for the element an action targeted.

    ``locator`` is a ``command`` string (evaluated as ``page.<locator>``).
    ``match_count`` is how many elements it matched on the live page right before
    the action ran; ``None`` means the count could not be evaluated.
    """

    locator: str
    kind: str
    score: int
    match_count: int | None = None

    @property
    def is_unique(self) -> bool:
        return self.match_count == 1


class CachedElement(BaseModel):
    """The DOM node behind the action's index, as browser-use resolved it."""

    tag: str
    attributes: dict[str, str] = Field(default_factory=dict)
    role: str | None = None
    name: str | None = None
    text: str | None = None
    xpath: str | None = None
    frame_id: str | None = None


class CachedAction(BaseModel):
    """One browser-use action, recorded around its execution in ``multi_act``."""

    step: int
    action_index: int
    action_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    url_before: str | None = None
    url_after: str | None = None
    element: CachedElement | None = None
    locators: list[CachedLocator] = Field(default_factory=list)
    error: str | None = None
    is_done: bool = False
    success: bool | None = None
    extracted_content: str | None = None
    started_download: bool = False
    duration_s: float | None = None
    category: ActionCategory | None = None
    reason: str | None = None

    @property
    def best_locator(self) -> CachedLocator | None:
        """Highest-scoring candidate that matched exactly one element."""
        return next((loc for loc in self.locators if loc.is_unique), None)


class ActionCache(BaseModel):
    """Every action an ``agentic_task`` run took, in execution order."""

    task: str
    actions: list[CachedAction] = Field(default_factory=list)
