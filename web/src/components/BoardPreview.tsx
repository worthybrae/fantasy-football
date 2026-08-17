import { useEffect, useState } from 'react'
import { fetchLandingPreview, type LandingPlayer } from '../api'

// The landing page's proof panel: the top of the board, drawn the way the
// live board draws it, from this machine's own DuckDB. Not a screenshot --
// there is no hosted demo data to fake, and the viewer's own board is more
// convincing than anyone else's anyway.

const PICK_SECONDS = 59

// A draft is a countdown; that is the whole feeling the live board exists
// for. The clock here is a demo (the panel says PREVIEW), so it just loops
// -- but it is the one thing on a static page that reads as "live".
function useDemoClock(): number {
  const [seconds, setSeconds] = useState(47)

  useEffect(() => {
    // A ticking clock is motion. Someone who has asked the OS for less of it
    // gets a stopped clock, which still reads as a draft board.
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (reduced?.matches) return
    const id = setInterval(
      () => setSeconds((s) => (s <= 0 ? PICK_SECONDS : s - 1)),
      1000,
    )
    return () => clearInterval(id)
  }, [])

  return seconds
}

function clockLabel(seconds: number): string {
  return `0:${String(seconds).padStart(2, '0')}`
}

function PreviewRow({ player }: { player: LandingPlayer }) {
  return (
    <li className="preview-row">
      <span className="preview-rank mono">{player.rank}</span>
      <span className="preview-name">{player.name}</span>
      <span className={`pos-badge pos-badge-${player.position.toLowerCase()}`}>
        {player.position}
      </span>
      <span className="preview-team mono">{player.team ?? '—'}</span>
      <span className="preview-tier mono">{player.tier === null ? '—' : `T${player.tier}`}</span>
      <span className="preview-vor mono">{player.vor === null ? '—' : player.vor.toFixed(1)}</span>
    </li>
  )
}

export default function BoardPreview({ limit = 8 }: { limit?: number }) {
  const [players, setPlayers] = useState<LandingPlayer[] | null>(null)
  const [pool, setPool] = useState<number | null>(null)
  const [failed, setFailed] = useState(false)
  const seconds = useDemoClock()

  useEffect(() => {
    let cancelled = false
    fetchLandingPreview(limit)
      .then((body) => {
        if (cancelled) return
        setPlayers(body.players)
        setPool(body.pool)
      })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => { cancelled = true }
  }, [limit])

  return (
    <section className="preview" aria-labelledby="preview-heading">
      <header className="preview-head">
        <h2 className="preview-eyebrow" id="preview-heading">
          Preview — your board, on the clock
        </h2>
        <span className="preview-clock mono" aria-hidden="true">
          PICK 1.04 <span className="preview-clock-time">{clockLabel(seconds)}</span>
        </span>
      </header>

      {failed && (
        <p className="preview-empty">
          The board preview needs the helper running. Start it with{' '}
          <code>make up</code>, then reload.
        </p>
      )}

      {!failed && players === null && (
        // Reserves the finished panel's height so the page does not jump when
        // the board build (seconds, not milliseconds) finally lands.
        <ul className="preview-rows" aria-hidden="true">
          {Array.from({ length: limit }, (_, i) => (
            <li className="preview-row preview-row-skeleton" key={i} />
          ))}
        </ul>
      )}

      {players !== null && (
        <ul className="preview-rows">
          {players.map((p) => <PreviewRow key={p.rank} player={p} />)}
        </ul>
      )}

      {pool !== null && (
        <p className="preview-foot mono">
          {pool} players scored · tiers and value over replacement from your own data
        </p>
      )}
    </section>
  )
}
