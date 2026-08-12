import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchProfile, playerSlug, setDrafted, type PlayerProfileData, type ScheduleWeek, type SeasonSummary } from '../api'

interface PlayerCardProps {
  playerId: string
  onClose: () => void
  // Avail%/ΔEV live on the board row, not the profile payload -- the draft
  // grid (Task 5) already has the full board list in hand when it opens
  // this card, so it passes the one row's values down instead of this card
  // re-fetching the whole board just to pick one out. Both default to null
  // so the card renders correctly when mounted with only a playerId -- e.g.
  // the grid's own data source, GET /api/sim/board, carries no per-player
  // `ev` at all, so it will mount this card with neither prop set.
  avail_pct?: number | null
  // Already relative to the board's best available option -- the same
  // `delta = p.ev - bestEv` PlayerTable.tsx computes for its ΔEV column, not
  // the raw `Player.ev`. The caller must do that subtraction before passing
  // it down; this card has no board list to compute bestEv from itself.
  // Because of that contract, 0 specifically means "tied for the board's
  // best," which is what justifies rendering it as a dash below -- a raw
  // absolute ev of 0 would have no such meaning and dashing it would make a
  // real (low) value indistinguishable from "no sim data."
  evDelta?: number | null
}

// duplicated from PlayerProfile.tsx (both are unexported there)
function depthSlotLabel(position: string, depthSlot: number | null): string | null {
  if (depthSlot === null) return null
  return `${position}${depthSlot}`
}

// duplicated from PlayerProfile.tsx -- higher sos_raw/sos_pct = opponents
// allow more fantasy points at this position = an easier ("softer")
// schedule; lower = a tougher one. Copied rather than shared -- two
// three-line consumers of a pure function don't earn a module of their own.
function sosLabel(sosRaw: number | null, sosPct: number | null): string {
  if (sosRaw === null || sosPct === null) return 'SoS —'
  const direction = sosPct >= 50 ? 'softer' : 'tougher'
  return `SoS ${sosRaw.toFixed(1)} FPA/g (${sosPct.toFixed(0)}th pct — ${direction})`
}

const fmt1 = (n: number | null) => (n === null ? '—' : n.toFixed(1))
// Edge (market_rank - rank) and the projection/actual gap both read in
// either direction -- an explicit sign makes "the model likes them more
// than the market" vs. the reverse, or "projected above/below what they've
// actually done," unambiguous at a glance.
const fmtSigned1 = (n: number | null) => (n === null ? '—' : `${n > 0 ? '+' : ''}${n.toFixed(1)}`)

// 0-100, higher = softer matchup (opponent allows more fantasy points at
// this position) -- interpolated continuously between the two status
// tokens the design system reserves for exactly this: --ok at the soft end,
// --fail at the tough end. Drives the SoS tile's value color, so that
// number and the strip below it read as one color language.
function sosTone(pct: number): string {
  const t = Math.max(0, Math.min(100, pct))
  return `color-mix(in srgb, var(--ok) ${t}%, var(--fail) ${100 - t}%)`
}

// Same tone, laid down as a low-alpha fill rather than solid text color --
// what the schedule strip's cells and its soft/tough legend keys both use,
// so the legend swatch matches the cells it's explaining.
function sosTint(pct: number): string {
  return `color-mix(in srgb, ${sosTone(pct)} 32%, transparent)`
}

function latestSeason(seasons: SeasonSummary[]): SeasonSummary | null {
  return seasons.length === 0
    ? null
    : seasons.reduce((latest, s) => (s.season > latest.season ? s : latest))
}

interface Tile {
  key: string
  label: string
  value: string
  title: string
}

// The card's six-or-fewer "critical numbers," ranked by how much they'd
// change a live draft decision and filtered to what actually applies to
// this player. A rookie or anyone with no NFL weekly history (also true of
// every DST, and every K under this league's scoring -- see profile.py's
// build_profile) has no `seasons` rows at all, so finish/role/snaps/GP
// silently drop out rather than rendering a null or a zero; SoS still shows
// because it comes from the team+position schedule, not this player's own
// history, so it's available even for a player who has never taken an NFL
// snap. A position that earns zero tiles (most DSTs) renders no tile grid
// at all instead of an empty box.
function buildTiles(profile: PlayerProfileData, position: string): Tile[] {
  const tiles: Tile[] = []
  const latest = latestSeason(profile.seasons)

  if (latest) {
    tiles.push({
      key: 'finish',
      label: 'Finish',
      value: `${position}${latest.pos_finish}`,
      title: `${latest.season} positional finish by total PPR points`,
    })
  }

  if (latest && latest.games > 0) {
    if (position === 'RB') {
      tiles.push({
        key: 'role',
        label: 'Car/gm',
        value: (latest.carries / latest.games).toFixed(1),
        title: `${latest.carries} carries over ${latest.games} games, ${latest.season} -- the workload number that matters most for an RB`,
      })
    } else if ((position === 'WR' || position === 'TE') && latest.target_share !== null) {
      tiles.push({
        key: 'role',
        label: 'Tgt share',
        value: `${Math.round(latest.target_share * 100)}%`,
        title: `${latest.targets} targets, ${latest.season} share of team pass targets`,
      })
    } else if (position === 'QB') {
      tiles.push({
        key: 'role',
        label: 'Att/gm',
        value: (latest.attempts / latest.games).toFixed(1),
        title: `${latest.attempts} pass attempts over ${latest.games} games, ${latest.season}`,
      })
    }
  }

  if (latest && latest.snap_share !== null && ['RB', 'WR', 'TE'].includes(position)) {
    tiles.push({
      key: 'snaps',
      label: 'Snaps',
      value: `${Math.round(latest.snap_share * 100)}%`,
      title: `Offensive snap share, ${latest.season}`,
    })
  }

  if (latest) {
    // Hardcoded /17 matches the regular-season length profile.py's own
    // proj_ppg divides by -- a played-15-of-17 line is the injury-risk
    // signal a chart can't compress into three seconds.
    tiles.push({
      key: 'gp',
      label: 'GP',
      value: `${latest.games}/17`,
      title: `Games played, ${latest.season}`,
    })
  }

  if (profile.outlook.sos_pct !== null && profile.outlook.sos_raw !== null) {
    tiles.push({
      key: 'sos',
      label: 'SoS',
      value: `${Math.round(profile.outlook.sos_pct)}%ile`,
      title: sosLabel(profile.outlook.sos_raw, profile.outlook.sos_pct),
    })
  }

  return tiles
}

function scheduleTitle(w: ScheduleWeek, position: string, bye: number | null): string {
  if (w.opponent === null) {
    return w.week === bye ? `Week ${w.week} — bye` : `Week ${w.week} — no game`
  }
  const base = `Week ${w.week} — ${w.home ? 'vs' : 'at'} ${w.opponent}`
  return w.fpa_pg !== null && w.pct !== null
    ? `${base} · ${w.fpa_pg.toFixed(1)} PPR/g allowed to ${position}s (${w.pct.toFixed(0)}th pct)`
    : base
}

// The card's most distinctive element (see the task brief): a full season
// of matchup difficulty in one row, no scrolling, no legend to read before
// it means something -- soft/tough is color, not text. Both `weeks` and
// `outlook.sos_pct`/`sos_raw` come from the team+position schedule rather
// than this player's own game log, so this renders for rookies exactly as
// it does for veterans; weekly_difficulty() (profile.py) returns an empty
// list for K/DST, so the strip -- like ScheduleCalendar's on the full
// profile page -- simply doesn't render for either.
function ScheduleStrip({ weeks, position, bye }: { weeks: ScheduleWeek[]; position: string; bye: number | null }) {
  if (weeks.length === 0) return null
  return (
    <div className="card-sched" aria-label={`${position} schedule difficulty by week`}>
      <div className="card-sched-strip">
        {weeks.map((w) => {
          const isBye = w.opponent === null && w.week === bye
          return (
            <span
              key={w.week}
              className={`card-sched-cell${isBye ? ' card-sched-bye' : ''} mono`}
              style={w.pct !== null ? { background: sosTint(w.pct) } : undefined}
              title={scheduleTitle(w, position, bye)}
            >
              {w.week}
            </span>
          )
        })}
      </div>
      <p className="card-footnote">
        <span className="card-sched-key" style={{ background: sosTint(90) }} /> soft{' '}
        <span className="card-sched-key" style={{ background: sosTint(10) }} /> tough
        {' · '}last season's PPR/g allowed to {position}s
      </p>
    </div>
  )
}

// The three-second, draft-night version of PlayerProfile: one board row's
// worth of decision-making context in a modal, instead of the full research
// page. Opened by clicking a grid cell (Task 5); closed by Escape, a
// backdrop click, or the close button. Every number on this card is picked
// because it can change whether you take the player now -- see the task
// brief for the full rationale of what made the cut and what didn't.
export default function PlayerCard({ playerId, onClose, avail_pct = null, evDelta = null }: PlayerCardProps) {
  const [profile, setProfile] = useState<PlayerProfileData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Optimistic override for the drafted toggle, null while it agrees with
  // whatever the profile fetch returned. PlayerProfile re-fetches the whole
  // profile after toggling; this card is the draft-night view, where a
  // round-trip between "he's gone" and the button changing is the cost the
  // toggle exists to avoid.
  const [draftedOverride, setDraftedOverride] = useState<boolean | null>(null)

  // Same guard PlayerProfile.tsx uses (see its playerIdRef comment for the
  // full rationale): mirrors the current `playerId` synchronously every
  // render, so a fetch that resolves after the id has already moved on
  // (open A, immediately click a comp/cell for B) can't clobber what's
  // already on screen for the new id.
  const playerIdRef = useRef(playerId)
  playerIdRef.current = playerId

  useEffect(() => {
    const forPlayerId = playerId
    setProfile(null)
    setError(null)
    setLoading(true)
    setDraftedOverride(null)
    fetchProfile(forPlayerId)
      .then((data) => {
        if (playerIdRef.current === forPlayerId) setProfile(data)
      })
      .catch((e) => {
        if (playerIdRef.current === forPlayerId) {
          setError(e instanceof Error ? e.message : 'Failed to load player profile')
        }
      })
      .finally(() => {
        if (playerIdRef.current === forPlayerId) setLoading(false)
      })
  }, [playerId])

  // Scoped to this card being mounted at all, and removed on unmount --
  // the grid page has no board-level keyboard handling of its own, but a
  // stray listener left behind after the card closes would still be a leak.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  // Named `player`, not `header`, so it doesn't shadow the literal <header>
  // JSX tag below (the brief's CSS targets `.player-card header`).
  const player = profile?.header
  const depthSlot = player ? depthSlotLabel(player.position, profile.outlook.depth_slot) : null
  const showSim = avail_pct !== null || evDelta !== null
  const drafted = draftedOverride ?? player?.drafted ?? false
  const tiles = profile && player ? buildTiles(profile, player.position) : []

  // proj_delta = proj_ppg - w_ppg (profile.py). A big positive gap means the
  // projection is asking you to bet on a level of production this player's
  // recent, recency-weighted actual output has never reached -- per the
  // brief, "the single most useful warning on a draft board" -- so it gets
  // the --fail treatment instead of the neutral tone a small gap gets. A
  // negative or modest gap isn't flagged the other way: an undersold
  // projection isn't a comparable risk, so it stays neutral rather than
  // inventing a "buy low" signal the data doesn't really support.
  const projDelta = profile?.summary.proj_delta ?? null
  const projDeltaWarn = projDelta !== null && projDelta >= 3

  async function handleToggleDrafted() {
    if (!player) return
    const next = !drafted
    setDraftedOverride(next)
    try {
      await setDrafted(player.player_id, next)
    } catch {
      // The board is the source of truth; if the write never landed, don't
      // leave the card claiming a pick that didn't happen.
      setDraftedOverride(!next)
    }
  }

  return (
    <div className="card-backdrop" onClick={onClose}>
      <div className="player-card" onClick={(e) => e.stopPropagation()}>
        {loading && !profile && <p>Loading…</p>}
        {error && <p className="error">{error}</p>}
        {player && profile && (
          <>
            <header>
              <h2>
                {player.name}
                {player.rookie && <span className="rookie-badge">R</span>}
              </h2>
              <span className="card-sub">
                <span className={`pos-badge pos-badge-${player.position.toLowerCase()}`}>
                  {player.position}
                </span>{' '}
                {player.team} · Bye {player.bye ?? '—'}
                {depthSlot && <> · {depthSlot}</>}
              </span>
              <button type="button" className="card-close" onClick={onClose} aria-label="Close">
                ✕
              </button>
            </header>

            {/* Headline: the single number a three-second decision hinges on
                most, per the brief -- how many points a week this player is
                projected for, next to what he's actually been putting up,
                so a projection that's outrunning his real production is
                visible before anything else on the card. */}
            <div className="card-headline">
              <div className="card-headline-primary">
                <span className="card-headline-value mono">{fmt1(profile.summary.proj_ppg)}</span>
                <span className="card-headline-label">Proj PPG</span>
              </div>
              <div className="card-headline-actual">
                <span className="card-headline-actual-value mono">{fmt1(profile.summary.w_ppg)}</span>
                <span className="card-headline-label">Recent PPG</span>
              </div>
              {projDelta !== null && (
                <span
                  className={`card-headline-delta${projDeltaWarn ? ' warn' : ''} mono`}
                  title="Projected PPG minus recency-weighted actual PPG"
                >
                  {fmtSigned1(projDelta)}
                </span>
              )}
            </div>

            <div className="card-row">
              <div>Rank <strong className="mono">{player.rank}</strong></div>
              <div>Tier <strong className="mono">{player.tier}</strong></div>
              <div>Mkt <strong className="mono">{fmt1(player.market_rank)}</strong></div>
              <div>Edge <strong className="mono">{fmtSigned1(player.edge)}</strong></div>
            </div>

            {showSim && (
              <div className="card-row card-sim">
                {avail_pct !== null && (
                  <div>Avail% <strong className="mono">{Math.round(avail_pct * 100)}%</strong></div>
                )}
                {evDelta !== null && (
                  <div>ΔEV <strong className="mono">{evDelta === 0 ? '—' : evDelta.toFixed(1)}</strong></div>
                )}
              </div>
            )}

            {tiles.length > 0 && (
              <div className="stat-tiles-row">
                {tiles.map((t) => (
                  <div className="stat-tile" key={t.key} title={t.title}>
                    <span
                      className="stat-tile-value mono"
                      style={t.key === 'sos' && profile.outlook.sos_pct !== null
                        ? { color: sosTone(profile.outlook.sos_pct) }
                        : undefined}
                    >
                      {t.value}
                    </span>
                    <span className="stat-tile-label">{t.label}</span>
                  </div>
                ))}
              </div>
            )}

            <ScheduleStrip weeks={profile.schedule} position={player.position} bye={profile.outlook.bye} />

            <div className="card-actions">
              <button
                type="button"
                className="drawer-draft-btn"
                onClick={handleToggleDrafted}
              >
                {drafted ? 'Undo draft' : 'Mark drafted'}
              </button>
              <Link to={`/legacy/players/${playerSlug(player.name)}`} className="card-full">
                Full profile →
              </Link>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
