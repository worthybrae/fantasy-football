import { useCallback, useState } from 'react'

// TWO VIEWS OF ONE ROOM, AND THE BROWSER REMEMBERS WHICH.
//
// Simple is the default for a fresh browser, deliberately: the room's whole
// redesign is that somebody drafting once a year should be told who to take
// and why, not handed fourteen columns and asked to rank them. The cheat
// sheet is not hidden and nothing in it was removed -- it is one click away,
// and once clicked it stays until it is clicked back.
//
// localStorage rather than a query string or an account setting: it is a
// preference about this browser's window, it has to survive the reload that
// follows a dropped socket, and it must never be worth a round trip on draft
// night.

export type RoomView = 'simple' | 'cheat'

const KEY = 'room-view'

/** Reads the stored preference. Anything other than the cheat sheet -- no
 *  value, a value from an older build, a browser that refuses storage
 *  entirely (Safari's private mode throws on the getter) -- is the simple
 *  view, which is what a fresh browser gets. */
function storedView(): RoomView {
  try {
    return window.localStorage.getItem(KEY) === 'cheat' ? 'cheat' : 'simple'
  } catch {
    return 'simple'
  }
}

/** The room's view, and the setter that persists it. Held in state as well as
 *  in storage so a switch repaints immediately; the write is best-effort and
 *  a failed one costs the reader the memory of his choice, never the choice
 *  itself. */
export function useRoomView(): [RoomView, (view: RoomView) => void] {
  const [view, setView] = useState<RoomView>(storedView)
  const choose = useCallback((next: RoomView) => {
    setView(next)
    try {
      window.localStorage.setItem(KEY, next)
    } catch {
      // Storage refused. The choice still holds for this session.
    }
  }, [])
  return [view, choose]
}

/** The switch itself. `aria-pressed` rather than a tab list: these are not
 *  tabs (the room already has a pair, Available and Snake Board, in the top
 *  bar) -- they are two settings of one control, and calling them tabs would
 *  give a screen reader two answers to "which tab am I on". */
export default function CheatSheetToggle({ view, onChange }: {
  view: RoomView
  onChange: (view: RoomView) => void
}) {
  return (
    <div className="room-view-toggle" role="group" aria-label="Draft room view">
      <button
        type="button"
        className={`room-view-btn${view === 'simple' ? ' is-active' : ''}`}
        aria-pressed={view === 'simple'}
        onClick={() => onChange('simple')}
      >
        Simple
      </button>
      <button
        type="button"
        className={`room-view-btn${view === 'cheat' ? ' is-active' : ''}`}
        aria-pressed={view === 'cheat'}
        onClick={() => onChange('cheat')}
      >
        Cheat sheet
      </button>
    </div>
  )
}
