# Draft Helper — ESPN Sync (browser extension)

One click on your ESPN draft tab syncs the draft into the Draft Helper. No
login, no password, no drag-a-bookmark step.

## What it sends, and what it doesn't

The extension reads your ESPN session **only** to fetch a per-draft token from
ESPN's own `draftSecurity` endpoint, and forwards **just that token** (plus the
public league and team ids) to the Draft Helper running on your machine.

- Sent: `leagueId`, `teamId`, `swid`, and the per-draft `token`.
- **Never sent:** your `espn_s2` session cookie or your ESPN/Disney password.

The token is scoped to a single draft and is worthless once that draft ends.
The Draft Helper holds it in memory and never writes it to disk.

## Load it (development, unpacked)

Until this is on the Chrome Web Store, load it directly:

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked** and choose this `extension/` folder
4. Pin the ⚓ Draft Helper icon to your toolbar

Works in Chrome, Edge, Brave, and any Chromium browser as-is. Firefox needs a
one-line `manifest.json` tweak; Safari needs Xcode repackaging.

## Use it

1. Start the Draft Helper (`make up`) so it is listening on `localhost:8000`.
2. Open your ESPN draft room.
3. Click the ⚓ icon → **Sync this draft**.
4. Switch to the Draft Helper tab — your board is live.

## Where the endpoint lives

`http://localhost:8000` is hard-coded in `background.js` as `APP`. Change it
there if the helper runs elsewhere. A shipped build would read this from
extension options rather than a constant.
