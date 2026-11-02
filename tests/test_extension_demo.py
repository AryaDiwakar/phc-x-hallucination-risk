"""The extension's demo page must be readable by the extension.

The sandbox shipped in the repository used ``class="msg user"`` markup, which
matched *none* of the extension's history selectors.  The guard therefore ran
with an empty conversation and always scored the prompt as low risk, so the
one page that exists to demonstrate the extension could not demonstrate it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "extension" / "demo" / "chat.html"
CONTENT = ROOT / "extension" / "content.js"


@pytest.fixture(scope="module")
def demo_html():
    if not DEMO.exists():
        pytest.skip("demo page not present")
    return DEMO.read_text()


@pytest.fixture(scope="module")
def content_js():
    if not CONTENT.exists():
        pytest.skip("content script not present")
    return CONTENT.read_text()


def _selectors(js: str):
    block = re.search(r"const HISTORY_SELECTORS = \[(.*?)\];", js, re.S)
    assert block, "HISTORY_SELECTORS not found in content.js"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_demo_messages_carry_a_role_the_extension_understands(demo_html):
    """Every message in the sandbox must expose its role to the extension."""
    roles = re.findall(r'data-message-author-role="(\w+)"', demo_html)
    assert roles, (
        "the demo page's messages carry no data-message-author-role, so the "
        "extension cannot tell user turns from assistant turns"
    )
    assert "user" in roles and "assistant" in roles


def test_demo_markup_matches_one_of_the_extension_selectors(demo_html, content_js):
    """The sandbox markup must be covered, and covered safely."""
    selectors = _selectors(content_js)
    classes = set(re.findall(r'class="msg (user|assistant)"', demo_html))
    assert classes, "the demo page no longer uses the .msg markup"

    # the extension's own role attribute is the primary path
    assert any("data-message-author-role" in s for s in selectors)

    # any selector that targets the .msg class must match the whole class
    # token: a substring match would also hit ".message", and would over-match
    # on unrelated pages.
    for s in selectors:
        if "msg" not in s:
            continue
        assert "[class~='msg']" in s or s == ".msg", (
            f"the {s!r} selector substring-matches 'msg', which also appears "
            "inside 'message' and would over-match"
        )


def test_extension_mentions_every_action_the_policy_can_emit(content_js):
    """The panel must understand every action the policy can return."""
    for action in ("CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN"):
        assert action in content_js, f"{action} is not handled by the panel"


def test_extension_collects_all_loaded_history_without_per_turn_truncation(content_js):
    """The ChatGPT integration must pass the full loaded transcript to PHC-X."""
    assert "function collectHistory()" in content_js
    assert "nodes.slice(-limit)" not in content_js
    assert "text.slice(0, 1200)" not in content_js


def test_extension_exposes_user_controlled_prompt_enhancement(content_js):
    assert 'improve.textContent = "Improve prompt"' in content_js
    assert 'type: "phc-enhance"' in content_js
    assert 'useDraft.textContent = "Use draft"' in content_js
