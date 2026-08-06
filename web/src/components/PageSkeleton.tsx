// Shimmering placeholder mirroring the board's table, shown on first load.
export function BoardSkeleton({ rows = 12 }: { rows?: number }) {
  return (
    <div className="board-skeleton" aria-busy="true" aria-label="Loading players">
      <div className="skeleton sk-row sk-head" />
      {Array.from({ length: rows }, (_, i) => (
        <div className="skeleton sk-row" key={i} />
      ))}
    </div>
  )
}

// Shimmering placeholder mirroring the player page's dossier grid, shown
// while the roster/profile requests are in flight. Animation is disabled
// under prefers-reduced-motion (see App.css).
export default function PageSkeleton() {
  return (
    <div className="player-page" aria-busy="true" aria-label="Loading player">
      <div className="skeleton sk-back" />
      <div className="skeleton sk-title" />
      <div className="skeleton sk-subtitle" />
      <div className="pp-grid">
        <div className="skeleton sk-block pp-span7" />
        <div className="skeleton sk-block pp-span5" />
        <div className="skeleton sk-chart pp-span7" />
        <div className="skeleton sk-chart pp-span5" />
        <div className="skeleton sk-block pp-span12" />
      </div>
    </div>
  )
}
