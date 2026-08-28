"""One size of face, asked for where the face leaves the process.

nflverse stores nfl.com's Cloudinary ORIGINAL for every player: 3400x2450
pixels, 250-870 KB apiece. Nothing this app draws is larger than a 44-pixel
avatar. A cold landing load measured 10.5 MB, and 10.2 MB of it was these
photographs -- every one of them downloaded whole and then scaled down by
the browser to something the size of a fingernail.

Cloudinary resizes on delivery. The segment after `/image/upload/` (or
`/image/private/` -- nflverse's rows carry both) is a comma-separated list
of transform parameters, and adding `w_96,c_fill,g_face` to it asks the CDN
for a 96-pixel square cropped around the face. Same URL shape, same
`f_auto,q_auto` the stored URL already carries, a few KB instead of a few
hundred.

WHY THE STORED URL IS LEFT ALONE. The width is a property of the SLOT the
photo lands in, not of the photo: a social card wants 320 and a table row
wants 96, and a `players` table holding one of those two could never answer
for the other. So the transform goes on at the point of serving, where the
size is known, and `pipeline/sources.py` keeps writing down what nflverse
published.

WHAT IS NOT A CLOUDINARY URL COMES BACK UNTOUCHED. Every fixture in the
suite, and any database whose photos came from somewhere else, holds plain
URLs with no transform segment to write into. Rewriting one would produce a
404 where there used to be a picture, so the test is "does this URL have a
Cloudinary delivery segment", and everything else is passed through as-is.
"""
from __future__ import annotations

import re

# The delivery-type segment, and the only thing here that identifies a URL
# as Cloudinary's. `private` is not a typo and not rare -- most of the
# nflverse headshots are served under it, and it takes transforms exactly
# the way `upload` does.
_DELIVERY = re.compile(r"/image/(?:upload|private|authenticated|fetch)/")

# One transform parameter: a short key, an underscore, a value. A transform
# segment is a comma-separated list of these. A public id or a version
# segment is not, which is what tells the two apart when a URL carries no
# transform at all (`/image/upload/league/xyz`, `/image/upload/v1/league`).
_PARAM = re.compile(r"^[a-z]{1,3}_[^,/]+$")

# The parameters this helper owns and therefore replaces rather than
# appends to. Two `w_` in one segment is undefined, and a crop mode left
# over from another width would fight the one being asked for. Replacing
# also makes the helper idempotent: thumb(thumb(u, 96), 320) is
# thumb(u, 320), which is what lets a caller hand a list avatar's URL to an
# og:image tag without having to have kept the original around.
_OWNED = ("w_", "h_", "c_", "g_", "ar_", "dpr_")

# What a face is asked for at. Every avatar in the app is drawn at 44 CSS
# pixels or less, so this is the 2x asset for the largest of them.
DEFAULT_WIDTH = 96


def thumb(url, width: int = DEFAULT_WIDTH) -> str | None:
    """`url` resized to `width`, if it is a URL a CDN will resize.

    None for anything that is not a non-empty string -- None itself, and
    the NaN a left-merged pandas column leaves behind for a player with no
    photo, both of which mean the same thing here: no face to draw.
    """
    if not isinstance(url, str) or not url:
        return None
    match = _DELIVERY.search(url)
    if match is None:
        return url

    head, tail = url[:match.end()], url[match.end():]
    if not tail:
        return url
    ask = f"w_{int(width)},c_fill,g_face"

    # The first segment after the delivery type is a transform only if
    # every comma-separated piece of it looks like one -- and only if
    # something follows it, since a URL ending there has no public id and
    # is not one this helper should be taking apart.
    first, slash, rest = tail.partition("/")
    parts = first.split(",")
    if slash and rest and all(_PARAM.match(p) for p in parts):
        kept = [p for p in parts if not p.startswith(_OWNED)]
        return f"{head}{','.join(kept + [ask])}/{rest}"
    return f"{head}{ask}/{tail}"
