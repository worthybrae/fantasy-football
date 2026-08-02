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
  onToggleDrafted: (p: Player) => void
}

const fmt1 = (n: number) => n.toFixed(1)
const fmtNullable = (n: number | null) => (n === null ? '—' : n)
const NUMERIC_COLUMNS = new Set(['rank', 'tier', 'bye', 'vor', 'composite', 'adp', 'edge'])

export default function PlayerTable({ players, onToggleDrafted }: PlayerTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'rank', desc: false }])

  const columns = useMemo<ColumnDef<Player>[]>(
    () => [
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
    []
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
          // Tier boundaries only make sense when the board is in rank order;
          // sorting by another column would scatter tiers non-contiguously.
          const prevRow = i > 0 ? rows[i - 1] : null
          const isTierBoundary =
            sorting[0]?.id === 'rank' &&
            prevRow !== null &&
            prevRow.original.tier !== row.original.tier
          return (
            <tr
              key={row.id}
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
          )
        })}
      </tbody>
    </table>
  )
}
