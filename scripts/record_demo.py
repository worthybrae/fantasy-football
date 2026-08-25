"""Re-record the desktop-only page's demo video against the dev server.

MobileGate.tsx shows web/public/demo.webm and demo-poster.jpg to phones in
place of the app. They are a recording of the real landing room, so they go
stale when the room changes: run `make up`, then

    .venv/bin/python scripts/record_demo.py

and commit the two files it rewrites. Roughly 30 seconds, ~2.5 MB.
"""
import asyncio, glob, shutil, os
from playwright.async_api import async_playwright
import pathlib
S=str(pathlib.Path(__file__).resolve().parent.parent / "web" / "public")
V=S+"/.demo-rec"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width":1280,"height":800}, record_video_dir=V, record_video_size={"width":1280,"height":800})
        pg = await ctx.new_page()
        await pg.goto("http://localhost:5173/?signedout=1", wait_until="networkidle")
        await pg.wait_for_timeout(1500)
        await pg.click("button.wc-watch")
        await pg.wait_for_timeout(800)
        await pg.screenshot(path=f"{S}/demo-poster.jpg", type="jpeg", quality=82)
        # hover top picks, rows
        for sel in [".avail-table tbody tr:nth-child(2)", ".avail-table tbody tr:nth-child(4)", ".avail-table tbody tr:nth-child(6)"]:
            try: await pg.hover(sel, timeout=2000)
            except Exception as e: print("hover fail", sel, e)
            await pg.wait_for_timeout(1500)
        await pg.wait_for_timeout(5000)
        try:
            await pg.click("button.draft-tab:has-text('Snake')", timeout=2000)
            await pg.wait_for_timeout(5000)
            await pg.click("button.draft-tab:has-text('Available')", timeout=2000)
        except Exception as e: print("tab fail", e)
        await pg.wait_for_timeout(8000)
        await ctx.close()
        await b.close()
    f = glob.glob(f"{V}/*.webm")[0]
    shutil.copy(f, f"{S}/demo.webm")
    shutil.rmtree(V)
    print("video", os.path.getsize(f"{S}/demo.webm"))
asyncio.run(main())
