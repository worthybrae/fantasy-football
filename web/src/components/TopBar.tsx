import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

interface TopBarProps {
  search: string
  onSearch: (s: string) => void
  /** Right-hand slot -- App passes <FreshnessBadge /> here so TopBar stays
   *  a plain layout shell rather than owning that fetch itself. */
  meta?: ReactNode
  /** True while the profile drawer is open. App owns selectedPlayerId, not
   *  TopBar, so it's threaded in as a prop -- the `/` shortcut is inert in
   *  that state since the search box it would focus is hidden behind the
   *  drawer. */
  searchShortcutDisabled?: boolean
}

export default function TopBar({ search, onSearch, meta, searchShortcutDisabled }: TopBarProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  // Document-level so `/` works no matter where focus currently is on the
  // board -- except while the user is already typing somewhere (this input
  // included; browsers report it as INPUT too), where a literal "/" should
  // just be typed, or while the profile drawer is open (searchShortcutDisabled),
  // where the search box it would focus isn't even visible.
  useEffect(() => {
    function onKeyDown(e: globalThis.KeyboardEvent) {
      if (searchShortcutDisabled) return
      if (e.key !== '/') return
      const target = e.target as HTMLElement | null
      const tag = target?.tagName
      const isTyping = tag === 'INPUT' || tag === 'TEXTAREA' || target?.isContentEditable
      if (isTyping) return
      e.preventDefault()
      inputRef.current?.focus()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [searchShortcutDisabled])

  function handleInputKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key !== 'Escape') return
    // Stop this from also reaching PlayerProfile's own window-level Esc
    // handler (which closes the drawer) -- this handler only ever runs
    // while the search input itself has focus, so scoping it here is what
    // keeps the two Esc behaviors from fighting when both are "live" at
    // once (drawer open, search focused).
    e.stopPropagation()
    onSearch('')
    inputRef.current?.blur()
  }

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <span className="topbar-dot" aria-hidden="true" />
        <h1 className="topbar-title">Draft Board</h1>
      </div>
      <div className="topbar-search">
        <input
          ref={inputRef}
          type="text"
          className="search-input"
          placeholder="Search players…  /"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          onKeyDown={handleInputKeyDown}
          aria-label="Search players by name or team"
        />
      </div>
      <nav className="top-nav">
        <NavLink to="/" end>Board</NavLink>
        <NavLink to="/draft-board">Grid</NavLink>
      </nav>
      <div className="topbar-meta">{meta}</div>
    </header>
  )
}
