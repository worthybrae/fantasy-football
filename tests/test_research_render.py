"""The lab's rule is that a number in a post is a number that was measured.

A chart is prose too. `_signed` exists because `_bars` draws `abs(v)`, so a
correlation of -0.46 and one of +0.46 came out as the same picture -- and in
E006 that is the difference between "this manager likes running backs" and
"he already has three". A sign bug here publishes a false claim silently,
which is the one failure mode the lab is built to prevent.
"""
import re

from research.render import BAR_W, _chart_block, _signed


def _rects(svg: str):
    # `-?` matters: a bar that overflows its chart is drawn at a NEGATIVE x,
    # so a digits-only pattern silently drops the very bar a scaling bug
    # produces and the assertion passes over the broken output.
    return [(int(m.group(1)), int(m.group(2)))
            for m in re.finditer(r'<rect x="(-?\d+)"[^>]*width="(\d+)"', svg)]


def test_signed_bars_grow_the_other_way_for_a_negative():
    svg = _signed([("up", [0.4]), ("down", [-0.4])], [], 0.5, "")
    (up_x, up_w), (down_x, down_w) = _rects(svg)

    assert up_w == down_w, "equal magnitudes must draw equal-length bars"
    # The positive starts at the zero line; the negative ENDS there.
    assert down_x + down_w == up_x
    assert down_x < up_x


def test_signed_labels_a_negative_on_its_own_side_of_zero():
    """The number sits outside the bar, so a negative reads left of zero
    instead of colliding with the axis it is measured against."""
    svg = _signed([("down", [-0.4])], [], 0.5, "")
    text = re.search(r'<text x="(\d+)"[^>]*text-anchor="(\w+)"', svg)
    (rect_x, rect_w), = _rects(svg)

    assert text.group(2) == "end"
    assert int(text.group(1)) <= rect_x


def test_signed_scales_against_the_largest_magnitude_not_the_largest_value():
    """A chart whose biggest number is NEGATIVE must still fit inside itself.

    Goes through `_chart_block` rather than calling `_signed` with a `vmax`
    already worked out, because the scaling is the thing under test: taking
    `max(v)` instead of `max(abs(v))` here yields a ceiling of 0.108 for a
    -0.5 bar, which draws four times wider than the chart. E006's most
    significant findings are both negative, so this is its normal case.
    """
    svg = _chart_block('::signed caption="t"\na|{neg}\nb|{pos}\n::',
                       {"neg": -0.5, "pos": 0.1})
    widths = [w for _, w in _rects(svg)]

    assert max(widths) <= BAR_W // 2, "a bar overflowed the chart"
    assert max(widths) > BAR_W // 4, "the largest bar should fill most of its side"
