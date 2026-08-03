import { useEffect, useMemo, useRef, useState } from 'react'
import {
  type ColumnDef,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table'
import type { Player } from '../api'

interface PlayerTableProps {
  players: Player[]
  onToggleDrafted: (p: Player) => Promise<void>
  onSelectPlayer: (p: Player) => void
  /** Tier is computed per-position, so the boundary rule is only meaningful
   *  when a single position is in view -- on ALL/FLEX, adjacent rows are
   *  usually different positions with unrelated tier numbers, so the rule
   *  would fire almost everywhere and just be noise. */
  showTierBreaks: boolean
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
// FFC/ESPN ranks are always whole numbers; FP's ECR can carry a decimal --
// show it only when present so the tooltip doesn't print "12.0".
const fmtSource = (n: number | null) => (n === null ? '—' : Number.isInteger(n) ? String(n) : n.toFixed(1))
const NUMERIC_COLUMNS = new Set(['rank', 'tier', 'bye', 'vor', 'composite', 'market_rank', 'edge'])

// |edge| < 3 is inside the market's normal rank-vs-rank noise for this board
// -- not a real signal either way, so it reads as neutral rather than a
// weak green/red that would imply more confidence than the number carries.
function edgeClass(v: number | null): string {
  if (v === null || Math.abs(v) < 3) return 'edge-neutral'
  return v > 0 ? 'edge-pos' : 'edge-neg'
}

export default function PlayerTable({
  players,
  onToggleDrafted,
  onSelectPlayer,
  showTierBreaks,
  selectedIndex,
  onVisibleRowsChange,
}: PlayerTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'rank', desc: false }])
  const tableRef = useRef<HTMLTableElement>(null)

  // Position-max VOR for the micro-bars, computed over whatever's currently
  // passed in (already search/tab/hide-drafted filtered by App) so the bars
  // stay scaled to what's actually on screen rather than the whole board.
  const maxVorByPosition = useMemo(() => {
    const m = new Map<string, number>()
    for (const p of players) {
      if (p.vor > (m.get(p.position) ?? -Infinity)) m.set(p.position, p.vor)
    }
    return m
  }, [players])

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
      { accessorKey: 'rank', header: 'Rank' },
      { accessorKey: 'tier', header: 'Tier' },
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
        accessorKey: 'vor',
        header: 'VOR',
        cell: ({ row }) => {
          const v = row.original.vor
          const max = maxVorByPosition.get(row.original.position) ?? 0
          const pct = max > 0 ? Math.max(0, Math.min(100, (v / max) * 100)) : 0
          return (
            <span className="vor-cell">
              <span className="vor-bar" style={{ width: `${pct}%` }} aria-hidden="true" />
              <span className="vor-value">{fmt1(v)}</span>
            </span>
          )
        },
      },
      {
        accessorKey: 'composite',
        header: 'Composite',
        cell: ({ getValue }) => fmt1(getValue<number>()),
      },
      {
        accessorKey: 'market_rank',
        header: 'Mkt',
        cell: ({ row }) => {
          const p = row.original
          if (p.market_rank === null) return '—'
          const { ffc, espn, fp, fp_tier } = p.market_sources
          const title = `FFC ${fmtSource(ffc)} · ESPN ${fmtSource(espn)} · FP ${fmtSource(fp)} · FP tier ${fmtSource(fp_tier)}`
          const showSpread = p.market_spread !== null && p.market_spread >= 12
          return (
            <span title={title}>
              {fmt1(p.market_rank)}
              {showSpread && (
                <span style={{ opacity: 0.6 }}> ±{Math.round((p.market_spread as number) / 2)}</span>
              )}
            </span>
          )
        },
      },
      {
        accessorKey: 'edge',
        header: 'Edge',
        cell: ({ getValue }) => {
          const v = getValue<number | null>()
          if (v === null) return '—'
          const sign = v > 0 ? '+' : ''
          return <span className={`edge-chip ${edgeClass(v)}`}>{sign}{fmt1(v)}</span>
        },
      },
    ],
    [onToggleDrafted, maxVorByPosition]
  )

  const table = useReactTable({
    data: players,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  const rows = table.getRowModel().rows

  // Tier boundaries only make sense when the board is in rank order within a
  // single position -- sorting by another column would scatter tiers
  // non-contiguously, and on ALL/FLEX views tier numbers reset per position
  // so the rule would fire on nearly every row. One pass over the
  // already-sorted rows, toggling on each tier change, rather than
  // per-row lookback -- same result, one pass; memoized so it only
  // recomputes when the row order or the gate itself actually changes,
  // not on every unrelated App/PlayerTable re-render.
  const tierBandGate = showTierBreaks && sorting[0]?.id === 'rank'
  const tierBandFlags = useMemo(() => {
    if (!tierBandGate) return []
    const flags: boolean[] = []
    let bandOn = false
    rows.forEach((row, i) => {
      if (i > 0 && rows[i - 1].original.tier !== row.original.tier) bandOn = !bandOn
      flags.push(bandOn)
    })
    return flags
  }, [rows, tierBandGate])

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
            tierBandGate && tierBandFlags[i] && 'tier-band',
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
