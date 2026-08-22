import { ageLabel } from '../../api'
import PopCard from './PopCard'
import { ATTR_EXACT, type NewsItem } from './payload'

// The two newest headlines, and how many there are. Newest first, exactly as
// the payload ordered them (see scoring/profile.py's `player_news`: recency
// is the primary key and attribution only the tiebreak).
//
// Two, not eight: this card sits in a row beside four comparable seasons and
// the popup is 534px wide. The count in the head is what says there is more,
// and the headline itself is a link to the story.
const HEADLINES = 2

// The two feeds are kept visibly apart, as they were when this was a full
// section. An ESPN-tagged item is about this player because ESPN said so; a
// name-matched item is about this player because a search for his name and
// team returned it, which measured 93-96% relevant and is therefore right
// most of the time and wrong sometimes. In a card this size the distinction
// is a filled mark against a hollow one -- shape, not colour alone -- and
// the link's own title says which in words.
const ATTR_TITLE = {
  exact: 'ESPN tagged this article with his athlete id',
  match: 'Matched by searching his name and team — usually him, not always',
}

export default function NewsPanel({ items }: { items: NewsItem[] }) {
  // Empty for a player nobody wrote about, and for every defense. A "News"
  // card with a dash in it is a promise the payload cannot keep.
  if (items.length === 0) return null

  return (
    <PopCard title="News" note={`${items.length} ${items.length === 1 ? 'story' : 'stories'}`}>
      <ul className="pp-pop-news">
        {items.slice(0, HEADLINES).map((item) => {
          const tagged = item.attribution === ATTR_EXACT
          return (
            <li className="pp-pop-news-item" key={item.url}>
              <span
                className={`pp-pop-news-dot${tagged ? ' is-exact' : ''}`}
                aria-hidden="true"
              />
              <div className="pp-pop-news-body">
                <a
                  className="pp-pop-news-link"
                  href={item.url}
                  target="_blank"
                  // noopener because this opens a page nobody here controls,
                  // and the tab it opens would otherwise get a handle on
                  // this one.
                  rel="noreferrer noopener"
                  title={tagged ? ATTR_TITLE.exact : ATTR_TITLE.match}
                >
                  {item.headline}
                </a>
                <div className="mono pp-pop-news-meta">
                  {item.source ?? 'unattributed'}
                  {item.published_at !== null && ` · ${ageLabel(item.published_at)}`}
                </div>
              </div>
            </li>
          )
        })}
      </ul>
    </PopCard>
  )
}
