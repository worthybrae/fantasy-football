import { ageLabel } from '../../api'
import { ATTR_EXACT, type NewsItem } from './payload'

// Recent headlines, newest first, exactly as the payload ordered them (see
// scoring/profile.py's `player_news`: recency is the primary key and
// attribution only the tiebreak).
//
// The two feeds are kept visibly apart. An ESPN-tagged item is about this
// player because ESPN said so; a name-matched item is about this player
// because a search for his name and team returned it, which measured
// 93-96% relevant and is therefore right most of the time and wrong
// sometimes. Rendering both as one undifferentiated list would present the
// second as the first -- and two players who share a name write their
// headlines out of the same words.
export default function NewsPanel({ items }: { items: NewsItem[] }) {
  if (items.length === 0) return null
  const newest = items.find((i) => i.published_at !== null)?.published_at ?? null
  const fetched = items.find((i) => i.fetched_at !== null)?.fetched_at ?? null

  return (
    <div className="pp-news">
      <div className="pp-section-meta mono">
        {newest !== null ? `newest ${ageLabel(newest)}` : 'undated'}
        {fetched !== null && <> · feed read {ageLabel(fetched)}</>}
      </div>
      <ul className="pp-news-list">
        {items.map((item) => (
          <li className="pp-news-item" key={item.url}>
            <a
              className="pp-news-link"
              href={item.url}
              target="_blank"
              // noopener because this opens a page nobody here controls, and
              // the tab it opens would otherwise get a handle on this one.
              rel="noreferrer noopener"
            >
              {item.headline}
            </a>
            <div className="pp-news-meta mono">
              {item.source ?? 'unattributed'}
              {item.published_at !== null && <> · {ageLabel(item.published_at)}</>}
              {item.attribution === ATTR_EXACT
                ? <span className="pp-news-tag is-exact" title="ESPN tagged this article with his athlete id">tagged</span>
                : <span className="pp-news-tag" title="Matched by searching his name and team — usually him, not always">name match</span>}
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
