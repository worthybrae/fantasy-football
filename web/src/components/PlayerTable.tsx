import { useMemo, useState } from 'react'
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
}

const fmt1 = (n: number) => n.toFixed(1)
const fmtNullable = (n: number | null) => (n === null ? '—' : n)
const NUMERIC_COLUMNS = new Set(['rank', 'tier', 'bye', 'vor', 'composite', 'adp', 'edge'])

export default function PlayerTable({ players, onToggleDrafted, onSelectPlayer, showTierBreaks }: PlayerTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'rank', desc: false }])

  const columns = useMemo<ColumnDef<Player>[]>(
    () => [
      {
        id: 'drafted-toggle',
        header: '',
        enableSorting: false,
        cell: ({ row }) => (
          <button
            type="button"
            className="drafted-toggle"
            title="toggle drafted"
            aria-label={row.original.drafted ? 'Mark undrafted' : 'Mark drafted'}
            onClick={(e) => {
              // Row itself opens the profile drawer on click -- without this
              // the button click would bubble up and open the drawer too.
              e.stopPropagation()
              onToggleDrafted(row.original)
            }}
          >
            ✓
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
    [onToggleDrafted]
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
          return (
            <tr
              key={row.id}
              onClick={() => onSelectPlayer(row.original)}
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
          )
        })}
      </tbody>
    </table>
  )
}
