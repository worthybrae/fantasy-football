import { useEffect, useMemo, useRef, useState } from 'react'
import {
  type ColumnDef,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table'
import { bestEv as boardBestEv, type Player } from '../api'
import { boardColumnsFor } from '../statColumns'

// Selectable sort/rank sources for the board's rank column. 'agg' is the
// blended market_rank; the rest read straight from market_sources.
export const RANK_SOURCES = [
  { id: 'agg', label: 'Aggregate', short: 'Mkt' },
  { id: 'ffc', label: 'FFC ADP', short: 'FFC' },
  { id: 'espn', label: 'ESPN PPR', short: 'ESPN' },
  { id: 'fp', label: 'FantasyPros', short: 'FP' },
  { id: 'mfl', label: 'MFL ADP', short: 'MFL' },
  { id: 'cbs', label: 'CBS', short: 'CBS' },
] as const
export type RankSourceId = (typeof RANK_SOURCES)[number]['id']

function rankValue(p: Player, source: RankSourceId): number | null {
  return source === 'agg' ? p.market_rank : p.market_sources[source]
}

interface PlayerTableProps {
  players: Player[]
  onToggleDrafted: (p: Player) => Promise<void>
  onSelectPlayer: (p: Player) => void
  /** Which ranking populates and sorts the rank column ('agg' = blended). */
  rankSource: RankSourceId
  /** Drives which position-specific stat columns are appended after
   *  the market columns -- ALL/FLEX/K/DST show the summary stats only. */
  positionFilter: string
  /** Keyboard-cursor row, as an index into the current sorted row order --
   *  owned by App so it survives this component re-rendering, and null
   *  means "no keyboard selection yet." */
  selectedIndex: number | null
  /** Fired whenever the VISIBLE (filtered+sorted) row order changes, so App
   *  can map arrow-key movement and Enter/D to the right player without
   *  duplicating TanStack's sort here. */
  onVisibleRowsChange: (ids: string[]) => void
}

const fmt1 = (n: number) => n.toFixed(1)
const fmtNullable = (n: number | null) => (n === null ? '—' : n)
// ESPN's PPR rank is always a whole number.
const fmtRank = (n: number | null) => (n === null ? '—' : String(Math.round(n)))
// FFC/ESPN ranks are always whole numbers; FP's ECR can carry a decimal --
// show it only when present so the tooltip doesn't print "12.0".
const fmtSource = (n: number | null) => (n === null ? '—' : Number.isInteger(n) ? String(n) : n.toFixed(1))
// The rank column's id embeds the selected source (rank_agg, rank_ffc, …).
// TanStack caches each row's accessor values BY column id, so reusing one id
// while swapping the accessor serves stale cached values to the sorter --
// the header and cells update but the row order doesn't. A per-source id
// gets a fresh cache slot instead.
const rankColumnId = (source: RankSourceId) => `rank_${source}`

const NUMERIC_COLUMNS = new Set([
  'bye', 'espn_ppr_rank', 'avail_pct', 'ev',
  ...RANK_SOURCES.map((s) => rankColumnId(s.id)),
  ...['QB', 'RB', 'WR'].flatMap((p) => boardColumnsFor(p).map((c) => c.id)),
])
// Column ids that exist regardless of positionFilter -- used below to decide
// whether a stale sort (e.g. sorted by an RB-only stat column) still applies
// after switching tabs.
const STATIC_COLUMN_IDS = [
  'drafted-toggle', 'espn_ppr_rank', 'name', 'position', 'team', 'bye',
  'avail_pct', 'ev',
  ...RANK_SOURCES.map((s) => rankColumnId(s.id)),
]

export default function PlayerTable({
  players,
  onToggleDrafted,
  onSelectPlayer,
  rankSource,
  positionFilter,
  selectedIndex,
  onVisibleRowsChange,
}: PlayerTableProps) {
  // Default order is the aggregated market rank -- the mean of every
  // PPR-native source (FFC/ESPN PPR/FP/MFL/CBS) -- not any single site.
  const [sorting, setSorting] = useState<SortingState>([{ id: rankColumnId('agg'), desc: false }])

  // Picking a rank source is a request to sort by it -- snap the sort back
  // to the rank column even if the user had sorted by some stat column.
  useEffect(() => {
    setSorting([{ id: rankColumnId(rankSource), desc: false }])
  }, [rankSource])
  const tableRef = useRef<HTMLTableElement>(null)

  // ΔEV (below) is rendered relative to the best EV on the board. Computed
  // from `players` (not the sorted table rows), so it doesn't shift as the
  // sort/filter changes -- and via the shared helper, so PlayerCard's own
  // ΔEV can't end up measured against a different baseline.
  const bestEv = useMemo(() => boardBestEv(players), [players])

  const columns = useMemo<ColumnDef<Player>[]>(
    () => [
      {
        id: 'drafted-toggle',
        header: '',
        enableSorting: false,
        cell: ({ row }) => {
          const drafted = row.original.drafted
          return (
            <button
              type="button"
              className="drafted-toggle"
              title={drafted ? 'Undo drafted' : 'Mark drafted'}
              aria-label={drafted ? 'Undo drafted' : 'Mark drafted'}
              onClick={(e) => {
                // Row itself opens the profile drawer on click -- without
                // this the button click would bubble up and open it too.
                e.stopPropagation()
                onToggleDrafted(row.original)
                // Blur so keyboard nav (App's board keydown effect) stays
                // live for the very next keypress instead of this button
                // holding focus. Pointer-only (`detail` is 0 for a
                // keyboard-synthesized click) -- Tab/Enter activation keeps
                // focus here by design, covered instead by App's Enter/D
                // activeElement guard, so blurring unconditionally would
                // have broken Tab order for keyboard users.
                if (e.detail > 0) e.currentTarget.blur()
              }}
            >
              {/* U+FE0E forces text presentation -- without it, this glyph
                  renders as a color emoji on some platform/font
                  combinations, which would look out of place next to a
                  plain-text ✓ in an otherwise monospace/terminal UI. */}
              {drafted ? '↩︎' : '✓'}
            </button>
          )
        },
      },
      {
        id: 'espn_ppr_rank',
        // TanStack's `sortUndefined` nulls-last handling only special-cases
        // `undefined`, not `null` -- map the API's `null` to `undefined`
        // here so ranks (which mostly come back present) sort correctly
        // without a bespoke sortingFn.
        accessorFn: (row) => row.espn_ppr_rank ?? undefined,
        header: 'ESPN PPR',
        sortUndefined: 'last',
        cell: ({ row }) => fmtRank(row.original.espn_ppr_rank),
      },
      {
        accessorKey: 'name',
        header: 'Name',
        cell: ({ row }) => (
          <>
            {row.original.name}
            {row.original.rookie && <span className="rookie-badge">R</span>}
          </>
        ),
      },
      {
        accessorKey: 'position',
        header: 'Pos',
        cell: ({ getValue }) => {
          const pos = getValue<string>()
          return <span className={`pos-badge pos-badge-${pos.toLowerCase()}`}>{pos}</span>
        },
      },
      { accessorKey: 'team', header: 'Team' },
      {
        accessorKey: 'bye',
        header: 'Bye',
        cell: ({ getValue }) => fmtNullable(getValue<number | null>()),
      },
      {
        id: rankColumnId(rankSource),
        // null -> undefined so sortUndefined 'last' applies (TanStack only
        // special-cases undefined), same as the espn_ppr_rank column.
        accessorFn: (row) => rankValue(row, rankSource) ?? undefined,
        header: RANK_SOURCES.find((s) => s.id === rankSource)?.short ?? 'Mkt',
        sortUndefined: 'last',
        cell: ({ row }) => {
          const p = row.original
          const value = rankValue(p, rankSource)
          if (value === null) return '—'
          const { ffc, espn, fp, mfl, cbs, fp_tier } = p.market_sources
          const title = `FFC ${fmtSource(ffc)} · ESPN PPR ${fmtSource(espn)} · FP ${fmtSource(fp)} · MFL ${fmtSource(mfl)} · CBS ${fmtSource(cbs)} · FP tier ${fmtSource(fp_tier)}`
          if (rankSource !== 'agg') {
            return <span title={title}>{fmtSource(value)}</span>
          }
          const showSpread = p.market_spread !== null && p.market_spread >= 12
          return (
            <span title={title}>
              {fmt1(value)}
              {showSpread && (
                <span style={{ opacity: 0.6 }}> ±{Math.round((p.market_spread as number) / 2)}</span>
              )}
            </span>
          )
        },
      },
      {
        id: 'avail_pct',
        header: () => (
          <span title="Probability this player is still available at your next pick">Avail%</span>
        ),
        // null -> undefined so sortUndefined 'last' applies, same as the
        // other nullable numeric columns above.
        accessorFn: (row) => row.avail_pct ?? undefined,
        sortUndefined: 'last',
        sortDescFirst: true,
        cell: ({ row }) => {
          const v = row.original.avail_pct
          // Blank (not "0%") when a sim hasn't run yet; an actual 0% still
          // renders, since "definitely gone" is a real, distinct answer.
          return v === null ? '' : `${Math.round(v * 100)}%`
        },
      },
      {
        id: 'ev',
        header: () => (
          <span title="Expected starting-lineup points versus the best available option">ΔEV</span>
        ),
        accessorFn: (row) => row.ev ?? undefined,
        sortUndefined: 'last',
        sortDescFirst: true,
        // Rendered relative to the best EV on the board, so the top
        // candidate reads as a dash and everything else reads as what it
        // costs you.
        cell: ({ row }) => {
          const p = row.original
          if (p.ev === null || bestEv === null) return ''
          const delta = p.ev - bestEv
          return delta === 0 ? '—' : delta.toFixed(1)
        },
      },
      ...boardColumnsFor(positionFilter).map((c): ColumnDef<Player> => ({
        id: c.id,
        // null stats (rookies/K/DST) -> undefined so sortUndefined applies
        accessorFn: (row) => (row.stats ? c.sortValue(row.stats) : undefined),
        header: c.label,
        sortUndefined: 'last',
        sortDescFirst: true,
        cell: ({ row }) => (row.original.stats ? c.cell(row.original.stats) : '—'),
      })),
    ],
    [onToggleDrafted, positionFilter, rankSource, bestEv]
  )

  // TanStack drops an active sort when its column disappears from the
  // column set (e.g. sorted RB by a rushing stat, then switched to ALL,
  // which has no rushing columns) -- rows silently fall back to raw payload
  // (VOR) order with no sort indicator. Reset to the default rank sort
  // whenever the sorted column id isn't in the new position's column set.
  useEffect(() => {
    const validIds = new Set([
      ...STATIC_COLUMN_IDS,
      ...boardColumnsFor(positionFilter).map((c) => c.id),
    ])
    const sortedId = sorting[0]?.id
    if (sortedId && !validIds.has(sortedId)) {
      setSorting([{ id: 'espn_ppr_rank', desc: false }])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [positionFilter])

  const table = useReactTable({
    data: players,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  const rows = table.getRowModel().rows

  // Report the visible sorted order up to App, which owns the keyboard
  // cursor and needs to map arrow-key movement / Enter / D to a player id
  // without re-deriving TanStack's sort itself. `rows` is referentially
  // stable across renders that don't change `players` or `sorting` (both
  // memoized/stateful upstream), so this only re-fires when the order
  // actually changes.
  useEffect(() => {
    onVisibleRowsChange(rows.map((r) => r.original.player_id))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows])

  // Keep the keyboard cursor in view as it moves past the edge of whatever's
  // currently scrolled into the table's viewport. Depends on `rows` too, not
  // just `selectedIndex` -- re-sorting while a cursor is active changes
  // which player sits at that index without changing the index itself, so
  // the row worth scrolling to can move even when this number doesn't.
  useEffect(() => {
    if (selectedIndex === null) return
    const el = tableRef.current?.querySelector<HTMLTableRowElement>(
      `tbody tr[data-row-index="${selectedIndex}"]`
    )
    el?.scrollIntoView({ block: 'nearest' })
  }, [selectedIndex, rows])

  return (
    <table ref={tableRef}>
      <thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th
                key={header.id}
                onClick={header.column.getToggleSortingHandler()}
                style={{ cursor: 'pointer' }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
                {{ asc: ' ↑', desc: ' ↓' }[header.column.getIsSorted() as string] ?? ''}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {rows.map((row, i) => {
          const classNames = [
            row.original.drafted && 'row-drafted',
            selectedIndex === i && 'row-selected',
          ]
            .filter(Boolean)
            .join(' ')
          return (
            <tr
              key={row.id}
              data-row-index={i}
              onClick={() => onSelectPlayer(row.original)}
              className={classNames || undefined}
            >
              {row.getVisibleCells().map((cell) => (
                <td
                  key={cell.id}
                  className={NUMERIC_COLUMNS.has(cell.column.id) ? 'mono' : undefined}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
