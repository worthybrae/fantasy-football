import type { DepthChartGroup } from '../api'

interface DepthChartCardProps {
  team: string
  groups: DepthChartGroup[]
}

// Offensive depth chart for the player's team, one column per position
// group, the profiled player highlighted. Data is the latest daily
// nflverse snapshot.
export default function DepthChartCard({ team, groups }: DepthChartCardProps) {
  if (groups.length === 0) return null
  return (
    <div className="depth-card">
      <div className="depth-columns">
        {groups.map((g) => (
          <div className="depth-col" key={g.position}>
            <span className={`pos-badge pos-badge-${g.position.toLowerCase()}`}>{g.position}</span>
            <ol className="depth-list">
              {g.players.map((p) => (
                <li key={`${g.position}-${p.rank}-${p.name}`}
                    className={p.is_me ? 'depth-player depth-me' : 'depth-player'}>
                  <span className="depth-rank mono">{p.rank}</span>
                  <span className="depth-name">{p.name}</span>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>
      <p className="card-footnote">{team} offense · latest depth chart</p>
    </div>
  )
}
