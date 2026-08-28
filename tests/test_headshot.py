"""What `thumb` may and may not do to a stored photo url.

The two properties that matter to a deployment are opposite ones: a
Cloudinary url MUST come back asking for a small file, and anything else
MUST come back untouched. Getting the second wrong turns every photo in a
fixture-shaped database into a 404, which is worse than the megabytes the
first one saves.
"""
from scoring.headshot import DEFAULT_WIDTH, thumb

NFL = "https://static.www.nfl.com/image"


def test_the_stored_transform_is_kept_and_the_size_is_added():
    """`f_auto,q_auto` is the format and quality negotiation nfl.com already
    asks for -- the width joins it rather than replacing it."""
    assert thumb(f"{NFL}/upload/f_auto,q_auto/league/abc123") == (
        f"{NFL}/upload/f_auto,q_auto,w_128,c_fill,g_face/league/abc123")


def test_the_private_delivery_type_is_rewritten_too():
    """Most of nflverse's headshots are served under `/image/private/`, not
    `/image/upload/`. It takes the same transforms, and a helper that only
    knew the documented one would have left the majority of the board
    downloading originals."""
    assert thumb(f"{NFL}/private/f_auto,q_auto/league/xyz") == (
        f"{NFL}/private/f_auto,q_auto,w_128,c_fill,g_face/league/xyz")


def test_a_url_with_no_transform_segment_gets_one():
    assert thumb(f"{NFL}/upload/league/abc123") == (
        f"{NFL}/upload/w_128,c_fill,g_face/league/abc123")


def test_a_version_segment_is_not_mistaken_for_a_transform():
    """`v1712345678` is a cache-busting version, and the transform has to go
    BEFORE it. Appending the width to it would produce a url Cloudinary
    reads as a public id."""
    assert thumb(f"{NFL}/upload/v1712345678/league/abc") == (
        f"{NFL}/upload/w_128,c_fill,g_face/v1712345678/league/abc")


def test_the_width_is_replaced_rather_than_repeated():
    """Which is what lets the same url be asked for twice at two sizes --
    the social card asks for 320 off the 128 the page is already using."""
    small = thumb(f"{NFL}/upload/f_auto,q_auto/league/abc")
    assert thumb(small, 320) == (
        f"{NFL}/upload/f_auto,q_auto,w_320,c_fill,g_face/league/abc")
    assert small.count("w_") == 1 and thumb(small, 320).count("w_") == 1


def test_a_crop_that_disagrees_with_the_new_width_is_dropped():
    assert thumb(f"{NFL}/upload/c_scale,h_400,w_400,q_auto/league/abc", 128) == (
        f"{NFL}/upload/q_auto,w_128,c_fill,g_face/league/abc")


def test_a_non_cloudinary_url_is_returned_exactly_as_it_came():
    """Every fixture in this suite, and any database whose photos came from
    somewhere else. There is nothing to rewrite and rewriting it would
    produce a 404 where a picture used to be."""
    for url in ("https://example.test/star.png",
                "http://x/1.png",
                "https://img.test/a/b/c.jpg",
                "https://static.www.nfl.com/league/abc"):
        assert thumb(url) == url


def test_no_photo_is_no_photo():
    """None, the NaN a left-merged pandas column leaves behind, and the
    empty string all mean the same thing to the frontend: draw nothing."""
    assert thumb(None) is None
    assert thumb(float("nan")) is None
    assert thumb("") is None


def test_the_width_asked_for_is_the_width_given():
    assert "w_320" in thumb(f"{NFL}/upload/f_auto,q_auto/league/abc", 320)
    assert "w_44" in thumb(f"{NFL}/upload/f_auto,q_auto/league/abc", 44)


def test_the_default_covers_the_biggest_slot_on_a_retina_screen():
    """52 CSS pixels is the largest avatar any of these lands in, and a 2x
    display asks for twice that. A default under it makes the biggest faces
    the blurry ones, which is the one place a reader would notice."""
    assert DEFAULT_WIDTH >= 2 * 52
    assert f"w_{DEFAULT_WIDTH}" in thumb(f"{NFL}/upload/f_auto,q_auto/league/abc")
