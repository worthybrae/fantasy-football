import { Fragment, useMemo, useState } from 'react'
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
  onToggleDrafted: (p: Player) => void
  /** Tier is computed per-position, so the boundary rule is only meaningful
   *  when a single position is in view -- on ALL/FLEX, adjacent rows are
   *  usually different positions with unrelated tier numbers, so the rule
   *  would fire almost everywhere and just be noise. */
  showTierBreaks: boolean
}

const fmt1 = (n: number) => n.toFixed(1)
const fmtNullable = (n: number | null) => (n === null ? '—' : n)
const NUMERIC_COLUMNS = new Set(['rank', 'tier', 'bye', 'vor', 'composite', 'adp', 'edge'])

const FACTORS: { key: keyof Player; label: string }[] = [
  { key: 'production', label: 'Production' },
  { key: 'durability', label: 'Durability' },
  { key: 'role', label: 'Role' },
  { key: 'environment', label: 'Environment' },
  { key: 'schedule', label: 'Schedule' },
]

export default function PlayerTable({ players, onToggleDrafted, showTierBreaks }: PlayerTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'rank', desc: false }])
  // Keyed by player_id rather than row index/id so it plays nicely with
  // sorting and refetches; resetting on refetch (a new Player[] identity) is
  // acceptable per spec, so no need to prune stale ids here.
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  function toggleExpanded(playerId: string) {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(playerId)) next.delete(playerId)
      else next.add(playerId)
      return next
    })
  }

  const columns = useMemo<ColumnDef<Player>[]>(
    () => [
      {
        id: 'expand',
        header: '',
        enableSorting: false,
        cell: ({ row }) => (
          <button
            type="button"
            className="expand-toggle"
            aria-label={expanded.has(row.original.player_id) ? 'Collapse factor breakdown' : 'Expand factor breakdown'}
            aria-expanded={expanded.has(row.original.player_id)}
            onClick={(e) => {
              // Row itself toggles "drafted" on click -- without this the
              // chevron click would bubble up and also (un)draft the player.
              e.stopPropagation()
              toggleExpanded(row.original.player_id)
            }}
          >
            {expanded.has(row.original.player_id) ? '▾' : '▸'}
          </button>
        ),
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
      { accessorKey: 'position', header: 'Pos' },
      { accessorKey: 'team', header: 'Team' },
      {
        accessorKey: 'bye',
        header: 'Bye',
        cell: ({ getValue }) => fmtNullable(getValue<number | null>()),
      },
      {
        accessorKey: 'vor',
        header: 'VOR',
        cell: ({ getValue }) => fmt1(getValue<number>()),
      },
      {
        accessorKey: 'composite',
        header: 'Composite',
        cell: ({ getValue }) => fmt1(getValue<number>()),
      },
      {
        accessorKey: 'adp',
        header: 'ADP',
        cell: ({ getValue }) => fmtNullable(getValue<number | null>()),
      },
      {
        accessorKey: 'edge',
        header: 'Edge',
        cell: ({ getValue }) => {
          const v = getValue<number | null>()
          if (v === null) return '—'
          const color = v > 0 ? 'green' : v < 0 ? 'red' : undefined
          return <span style={{ color }}>{fmt1(v)}</span>
        },
      },
    ],
    [expanded]
  )

  const table = useReactTable({
    data: players,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <table>
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
        {table.getRowModel().rows.map((row, i, rows) => {
          // Tier boundaries only make sense when the board is in rank order
          // within a single position -- sorting by another column would
          // scatter tiers non-contiguously, and on ALL/FLEX views tier
          // numbers reset per position so the rule would fire on nearly
          // every row.
          const prevRow = i > 0 ? rows[i - 1] : null
          const isTierBoundary =
            showTierBreaks &&
            sorting[0]?.id === 'rank' &&
            prevRow !== null &&
            prevRow.original.tier !== row.original.tier
          const isExpanded = expanded.has(row.original.player_id)
          return (
            <Fragment key={row.id}>
              <tr
                onClick={() => onToggleDrafted(row.original)}
                className={isTierBoundary ? 'tier-boundary' : undefined}
                style={
                  row.original.drafted
                    ? { opacity: 0.35, textDecoration: 'line-through', cursor: 'pointer' }
                    : { cursor: 'pointer' }
                }
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
              {isExpanded && (
                <tr className="factor-breakdown-row">
                  <td colSpan={row.getVisibleCells().length}>
                    <dl className="factor-breakdown">
                      {FACTORS.map(({ key, label }) => (
                        <div className="factor-breakdown-item" key={key}>
                          <dt>{label}</dt>
                          <dd className="mono">{fmt1(row.original[key] as number)}</dd>
                        </div>
                      ))}
                    </dl>
                  </td>
                </tr>
              )}
            </Fragment>
          )
        })}
      </tbody>
    </table>
  )
}
