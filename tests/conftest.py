import os

import pytest

# Importing the inference modules builds the task-runtime Settings, which requires
# these two values. Tests never talk to the Optexity API.
os.environ.setdefault("OPTEXITY_API_KEY", "test")
os.environ.setdefault("DEPLOYMENT", "dev")

from optexity.schema.action_cache import (  # noqa: E402
    CachedAction,
    CachedElement,
    CachedLocator,
)

PAGE = "https://example.com/form"


@pytest.fixture
def make_action():
    """Build a ``CachedAction`` the way ``ActionCacheRecorder`` would record it.

    Element actions get one locator candidate; ``unique=False`` makes it match two
    elements on the page, so the action has no ``best_locator``.
    """

    def make(
        name: str,
        params: dict | None = None,
        *,
        step: int = 1,
        index: int = 0,
        url: str = PAGE,
        url_after: str | None = None,
        element_name: str | None = "City",
        xpath: str = "html/body/form/input[1]",
        locator: str = 'get_by_role("textbox", name="City")',
        unique: bool = True,
        error: str | None = None,
        success: bool | None = None,
        started_download: bool = False,
    ) -> CachedAction:
        has_element = name in {"input", "click", "select_dropdown", "upload_file"}
        return CachedAction(
            step=step,
            action_index=index,
            action_name=name,
            params=params or {},
            url_before=url,
            url_after=url_after or url,
            element=(
                CachedElement(
                    tag="input", role="textbox", name=element_name, xpath=xpath
                )
                if has_element
                else None
            ),
            locators=(
                [
                    CachedLocator(
                        locator=locator,
                        kind="role+name",
                        score=72,
                        match_count=1 if unique else 2,
                    )
                ]
                if has_element
                else []
            ),
            error=error,
            is_done=name == "done",
            success=(
                success if success is not None else (True if name == "done" else None)
            ),
            started_download=started_download,
        )

    return make
