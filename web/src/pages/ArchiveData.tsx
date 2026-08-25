import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ageLabel, fetchMarketOverview, fetchMockDrafts, fetchUpcomingDrafts,
  type MarketOverview, type MockDraft, type UpcomingDrafts,
} from '../api'
import { readAccount, rememberAccount } from '../lib/accountCache'
import { useDocumentMeta } from '../lib/documentMeta'
import { Logo } from '../components/Logo'
import '../market.css'

// THE ARCHIVE'S RAW DATA: every recorded draft, one row each.
//
// /archive reads the corpus from one seat and draws conclusions. This page
// draws none. It is the list the conclusions are counted from, as a table a
// reader can scan, sort by eye and check -- which draft, when, how many
// picks, how many of the seats had a person in them -- and every row opens
// the board it summarises. The topbar, the gate and the stylesheet are the
// archive's own, so walking between the two pages changes only what is under
// the bar.

/** "3 of 8" -- the seats a person sat in. Null is not zero: a draft the farm
 *  recorded before it counted seats knows nothing about who was in the room. */
function humans(d: MockDraft): string {
  if (d.human_seats === null) return '—'
  return `${d.human_seats} of ${d.teams}`
}

// A page of cards, not an endless scroll: four hundred rows in one column is
// a list nobody reads past the fold of, and the browser laying out every one
// of them buys nothing. Two dozen a page keeps the grid one or two screens
// tall at every width.
const PER_PAGE = 24

export default function ArchiveData() {
  useDocumentMeta({
    title: 'The draft archive – ESPN Draft Assist',
    description: 'What hundreds of recorded ESPN mock drafts do from each seat: who is taken at your turn, and who is still there at the next one.',
  })
  const navigate = useNavigate()
  const [drafts, setDrafts] = useState<MockDraft[] | null>(null)
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  const [page, setPage] = useState(0)
  const [error, setError] = useState<string | null>(null)
  // The same gate as /archive, for the same reason: free, not anonymous.
  const [account, setAccount] = useState<UpcomingDrafts | null>(readAccount)

  useEffect(() => {
    let cancelled = false
    fetchMarketOverview()
      .then((body) => { if (!cancelled) setOverview(body) })
      .catch(() => { /* the facts strip simply does not render */ })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    fetchMockDrafts()
      .then((rows) => { if (!cancelled) setDrafts(rows) })
      .catch((err) => { if (!cancelled) setError(String(err.message || err)) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    fetchUpcomingDrafts()
      .then((body) => {
        rememberAccount(body)
        if (!cancelled) setAccount(body)
      })
      .catch(() => {
        if (!cancelled && account === null) setAccount({ connected: false, leagues: [] })
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const signedIn = account !== null && account.connected
  const live = drafts?.filter((d) => d.status === 'live').length ?? 0

  return (
    <div className="mk-page">
      <header className="draft-topbar">
        <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab" to="/">Drafts</Link>
          <Link className="draft-tab is-active" to="/archive" aria-current="page">Archive</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">
          {drafts === null
            ? 'Reading the archive…'
            : `${drafts.length} drafts${live > 0 ? ` · ${live} live` : ''}`}
        </span>
        <span className="draft-topbar-spacer" />
        <Link className="draft-topbar-link" to="/archive">← Back to the archive</Link>
      </header>

      <main className="mk" aria-busy={account === null}>
        {account !== null && !signedIn && (
          <section className="mk-panel mk-gate">
            <h2 className="mk-h2">The archive needs a signed-in account.</h2>
            <p className="mk-gate-copy">
              The draft archive is free. To prevent abuse, you can only read
              it with a connected ESPN account. Connect once with the Draft
              Assistant bookmark. Then come back to this page.
            </p>
            <button type="button" className="mk-gate-cta" onClick={() => navigate('/')}>
              Get set up
            </button>
          </section>
        )}

        {signedIn && (
          <>
            <header className="mkd-head">
              <div className="mkd-head-say">
                <h1 className="mk-title">The raw data</h1>
                <p className="mk-lede">
                  Every draft the archive is counted from, newest first. Open
                  any card to see its full board, pick by pick.
                </p>
              </div>
              {/* The whole corpus, to take away. A plain link to the zip the
                  server builds: drafts, picks, player names and the overview
                  as JSON files -- so a claim the page makes can be checked
                  without this page. */}
              <a className="mk-export" href="/api/market/export.zip">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 3v12" />
                  <path d="M6 11l6 6 6-6" />
                  <path d="M4 21h16" />
                </svg>
                Download the data
                <span className="mk-export-kind mono">.zip of JSON</span>
              </a>
            </header>

            {/* THE CORPUS, AS FOUR TILES. Each figure stands on its own
                card, the same footing as the drafts below them: the page is
                a field of cards, and the totals are the first row of it
                rather than a strip ruled off above it. */}
            {overview && (
              <dl className="mkd-stats">
                <div className="mkd-stat">
                  <dt>Drafts</dt>
                  <dd className="mono">{overview.drafts}</dd>
                </div>
                <div className="mkd-stat">
                  <dt>Picks</dt>
                  <dd className="mono">{overview.picks.toLocaleString()}</dd>
                </div>
                <div className="mkd-stat">
                  <dt>Made by people</dt>
                  <dd className="mono">
                    {overview.human_picks.toLocaleString()}
                    {overview.picks > 0 && (
                      <span className="mkd-stat-share">
                        {Math.round((overview.human_picks / overview.picks) * 100)}%
                      </span>
                    )}
                  </dd>
                </div>
                <div className="mkd-stat">
                  <dt>Live now</dt>
                  <dd className={`mono${live > 0 ? ' is-live' : ''}`}>{drafts === null ? '—' : live}</dd>
                </div>
                {overview.to && (
                  <div className="mkd-stat">
                    <dt>Newest</dt>
                    <dd className="mono">{ageLabel(overview.to)}</dd>
                  </div>
                )}
              </dl>
            )}

            {error && <p className="mk-error" role="alert">{error}</p>}

            {/* No panel around the cards: they float on the page, each its
                own object, the way the tiles above do. */}
            <section className="mkd-drafts" aria-label="Recorded drafts">
              {drafts === null && !error && <p className="mk-loading">Reading the archive…</p>}
              {drafts !== null && drafts.length === 0 && (
                <p className="mk-loading">No drafts recorded yet.</p>
              )}
              {drafts !== null && drafts.length > 0 && (() => {
                const pages = Math.max(1, Math.ceil(drafts.length / PER_PAGE))
                const at = Math.min(page, pages - 1)
                const from = at * PER_PAGE
                const shown = drafts.slice(from, from + PER_PAGE)
                return (
                  <>
                    <ol className="mkd-grid">
                      {shown.map((d) => {
                        const total = d.teams * d.rounds
                        const when = d.started_at ?? d.recorded_at
                        const pct = total > 0
                          ? Math.min(100, (d.picks_made / total) * 100) : 0
                        return (
                          <li key={d.id}>
                            <button
                              type="button"
                              className={`mkd-card${d.status === 'live' ? ' is-live' : ''}`}
                              onClick={() => navigate(`/mocks?draft=${encodeURIComponent(d.id)}`)}
                            >
                              <span className="mkd-card-top">
                                <span className={`mocks-pill ${d.status === 'live' ? 'is-live' : 'is-done'}`}>
                                  {d.status === 'live' && <span className="mocks-pill-dot" aria-hidden="true" />}
                                  {d.status === 'live' ? 'LIVE' : 'DONE'}
                                </span>
                                <span className="mkd-card-when"
                                      title={when === null ? undefined : new Date(when).toLocaleString()}>
                                  {when === null ? 'no date' : ageLabel(when)}
                                </span>
                              </span>
                              {/* No league id: it is ESPN's internal number, means nothing
                                  to a reader, and made every card open with a
                                  serial. The board behind the click names the
                                  room. */}
                              {/* The one figure that differs card to card
                                  while a room is playing: how far in it is. */}
                              <span className="mkd-card-picks">
                                <span className="mono mkd-card-n">{d.picks_made}</span>
                                <span className="mkd-card-of mono">/ {total}</span>
                                <span className="mkd-card-unit">picks</span>
                              </span>
                              <span className="mkd-card-bar" aria-hidden="true">
                                <span className="mkd-card-fill" style={{ width: `${pct}%` }} />
                              </span>
                              <span className="mkd-card-foot">
                                <span>{humans(d)} human</span>
                                <span className="mkd-card-open" aria-hidden="true">Board →</span>
                              </span>
                            </button>
                          </li>
                        )
                      })}
                    </ol>
                    {pages > 1 && (
                      <nav className="mkd-pager" aria-label="Pages of drafts">
                        <button type="button" className="mkd-page-btn"
                                disabled={at === 0}
                                onClick={() => setPage(at - 1)}>
                          ← Newer
                        </button>
                        <span className="mkd-page-where mono">
                          {from + 1}–{Math.min(from + PER_PAGE, drafts.length)} of {drafts.length}
                        </span>
                        <button type="button" className="mkd-page-btn"
                                disabled={at >= pages - 1}
                                onClick={() => setPage(at + 1)}>
                          Older →
                        </button>
                      </nav>
                    )}
                  </>
                )
              })()}
            </section>
          </>
        )}
      </main>
    </div>
  )
}
