"""The parts of the site's search presence that are files, not code.

Read straight off the source tree: a build step cannot lose what was never
in `web/index.html`, and a test that asserts on the built `dist` would be
skipped on every checkout without one."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
ROBOTS = ROOT / "web" / "public" / "robots.txt"


def test_the_document_head_describes_the_page():
    assert '<meta name="description"' in INDEX
    assert '<link rel="canonical" href="https://espnfantasydraft.com/"' in INDEX
    assert 'property="og:title"' in INDEX
    assert 'property="og:image" content="https://espnfantasydraft.com/demo-poster.jpg"' in INDEX
    assert 'name="twitter:card" content="summary_large_image"' in INDEX
    assert '<meta name="theme-color" content="#0a0b0e"' in INDEX


def test_the_document_carries_software_application_structured_data():
    assert 'application/ld+json' in INDEX
    assert '"@type": "SoftwareApplication"' in INDEX
    assert '"name": "ESPN Draft Assist"' in INDEX
    assert '"price": "9.99"' in INDEX


def test_robots_allows_the_site_and_hides_the_app_surfaces():
    text = ROBOTS.read_text(encoding="utf-8")
    assert "User-agent: *" in text
    for path in ("/api/", "/draft", "/live", "/room/"):
        assert f"Disallow: {path}" in text
    assert "Sitemap: https://espnfantasydraft.com/sitemap.xml" in text
