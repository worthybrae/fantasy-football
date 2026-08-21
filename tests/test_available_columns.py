"""The available table's headers must sit above the cells they name.

A static check on the .tsx, because there is no frontend test runner and this
is not a hypothetical: the Change column shipped with its header listed after
Health while its cell was rendered before, so for two commits every row drew
its change bar under the HEALTH heading and its health meter under CHANGE.
Nothing caught it -- the build passes, the lint passes, the types are fine,
and both cells render something plausible. Only a person looking at the screen
noticed, and what they saw was a health column that had become a thin line.

`scoring/board.py`'s column contract is pinned in test_board.py for the same
reason. This is that guarantee carried through to the thing a reader sees.
"""
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "web/src/components/draft/AvailableList.tsx"

# The sortable header key -> the token that must appear, in the same order,
# in the row body. A column whose cell is a bare value is matched on its own
# class, since it renders no component of its own.
COLUMN_RENDERS = {
    "finish": "FinishArc",
    "health": "HealthMeter",
    "steady": "SteadyMeter",
    "change": "ChangeMeter",
}

# Every column whose cell opens a hover panel. Each must be reachable, and
# none may still carry the native `title` it had before the panels existed --
# the OS box renders ON TOP of the panel, which is how a reader ended up with
# a tooltip covering the thing it was there to explain.
PANEL_KINDS = ("health", "games", "finish", "steady", "change")


def _source() -> str:
    return SOURCE.read_text()


def test_header_and_cell_order_agree_for_every_visual_column():
    src = _source()
    header_keys = re.findall(r"sortableTh\('(\w+)'", src)
    ordered = [k for k in header_keys if k in COLUMN_RENDERS]

    # Where each component is rendered in the row body, in document order.
    positions = {}
    for key, token in COLUMN_RENDERS.items():
        found = [m.start() for m in re.finditer(rf"<{token}\b", src)]
        assert found, f"{token} is not rendered anywhere -- did it get renamed?"
        positions[key] = found[0]

    by_body = sorted(ordered, key=lambda k: positions[k])
    assert by_body == ordered, (
        f"headers read {ordered} but the cells render {by_body} -- every "
        f"column between the first mismatch and the end is under the wrong "
        f"heading")


def test_every_visual_column_this_test_knows_about_still_exists():
    """Guards the guard: a renamed component would otherwise make the check
    above vacuous rather than failing."""
    src = _source()
    for key, token in COLUMN_RENDERS.items():
        assert f"sortableTh('{key}'" in src, f"header {key} vanished"
        assert f"<{token}" in src, f"component {token} vanished"


def test_every_panel_kind_is_reachable_from_a_cell():
    src = _source()
    for kind in PANEL_KINDS:
        assert f"onCellEnter('{kind}'" in src, f"nothing opens the {kind} panel"


def test_no_cell_that_opens_a_panel_still_carries_a_native_title():
    """A `title` on such a cell renders the OS tooltip over the panel."""
    src = _source()
    for meter in ("FinishArc", "HealthMeter", "SteadyMeter", "ChangeMeter"):
        start = src.index(f"const {meter} = memo(")
        body = src[start:src.index("})", start)]
        assert "title=" not in body, f"{meter} still sets a native title"
