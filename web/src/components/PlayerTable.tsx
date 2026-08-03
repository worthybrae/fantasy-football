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
import { boardColumnsFor } from '../statColumns'

interface PlayerTableProps {
  players: Player[]
  onToggleDrafted: (p: Player) => Promise<void>
  onSelectPlayer: (p: Player) => void
  /** Drives which position-specific stat columns (Task 3) are appended after
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
const NUMERIC_COLUMNS = new Set([
  'bye', 'market_rank', 'espn_ppr_rank',
  ...['QB', 'RB', 'WR'].flatMap((p) => boardColumnsFor(p).map((c) => c.id)),
])

export default function PlayerTable({
  players,
  onToggleDrafted,
  onSelectPlayer,
  positionFilter,
  selectedIndex,
  onVisibleRowsChange,
}: PlayerTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'espn_ppr_rank', desc: false }])
  const tableRef = useRef<HTMLTableElement>(null)

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
        accessorKey: 'market_rank',
        header: 'Mkt',
        cell: ({ row }) => {
          const p = row.original
          if (p.market_rank === null) return '—'
          const { ffc, espn, fp, fp_tier } = p.market_sources
          const title = `FFC ${fmtSource(ffc)} · ESPN ADP ${fmtSource(espn)} · FP ${fmtSource(fp)} · FP tier ${fmtSource(fp_tier)}`
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
    [onToggleDrafted, positionFilter]
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
