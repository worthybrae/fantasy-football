import { useState } from 'react'
import { ageLabel } from '../../api'
import PopCard from './PopCard'
import { ATTR_EXACT, type NewsItem } from './payload'
import { CARD_HINTS } from './hints'

// The newest headlines, and how many there are in total. Newest first,
// exactly as the payload ordered them (see scoring/profile.py's
// `player_news`: recency is the primary key and attribution only the
// tiebreak).
//
// Five at a time, and the rest a page away. It used to show six and stop,
// with the count in the head as the only sign that anything was behind it --
// which is a card telling a reader there are seventeen stories and then
// declining to show fourteen of them. The stories are already fetched; the
// only thing standing between a reader and the tenth headline was a number.
//
// Five rather than six because the pager takes a row of its own, and a card
// that grew by one line every time it gained a control would decide the
// height of its row for the card beside it. Paging is what the cap buys now,
// not concealment: the size is a layout choice, and no story is unreachable
// because of it.
const PER_PAGE = 5

// The two feeds are kept visibly apart, as they were when this was a full
// section. An ESPN-tagged item is about this player because ESPN said so; a
// name-matched item is about this player because a search for his name and
// team returned it, which measured 93-96% relevant and is therefore right
// most of the time and wrong sometimes.
//
// So EVERY item carries the word for its own feed. The first attempt at this
// card spent a 4px mark on it -- filled against hollow -- and hid the words
// in a `title`: a two-pixel difference nobody can see, behind a hover no
// keyboard and no screen reader ever performs. Under a pick clock this is
// the difference between a headline you trust and one you go and check, so
// it is text, on screen, on both.
const ATTR = {
  exact: { label: 'tagged', title: 'ESPN tagged this article with his athlete id' },
  match: {
    label: 'name match',
    title: 'Matched by searching his name and team — usually him, not always',
  },
}

export default function NewsPanel({ items }: { items: NewsItem[] }) {
  const [page, setPage] = useState(0)

  // Empty for a player nobody wrote about, and for every defense. A "News"
  // card with a dash in it is a promise the payload cannot keep.
  if (items.length === 0) return null

  // Clamped rather than reset, because nothing here resets it: click the
  // third page of a player with seventeen stories, then open one with four,
  // and the card would render an empty page of a list that has one. Clamping
  // on the way out means the state can be stale and the card cannot.
  const pages = Math.ceil(items.length / PER_PAGE)
  const at = Math.min(page, pages - 1)
  const start = at * PER_PAGE
  const shown = items.slice(start, start + PER_PAGE)

  return (
    <PopCard
      title="News"
      note={`${items.length} ${items.length === 1 ? 'story' : 'stories'}`}
      hint={CARD_HINTS.news}
    >
      <ul className="pp-pop-news">
        {shown.map((item) => {
          const tagged = item.attribution === ATTR_EXACT
          const attr = tagged ? ATTR.exact : ATTR.match
          return (
            <li className="pp-pop-news-item" key={item.url}>
              {/* A bullet, nothing more -- see `.pp-pop-news-dot`. */}
              <span className="pp-pop-news-dot" aria-hidden="true" />
              <div className="pp-pop-news-body">
                <a
                  className="pp-pop-news-link"
                  href={item.url}
                  target="_blank"
                  // noopener because this opens a page nobody here controls,
                  // and the tab it opens would otherwise get a handle on
                  // this one.
                  rel="noreferrer noopener"
                >
                  {item.headline}
                </a>
                <div className="mono pp-pop-news-meta">
                  <span>
                    {item.source ?? 'unattributed'}
                    {item.published_at !== null && ` · ${ageLabel(item.published_at)}`}
                  </span>
                  {/* The `title` is the long form for a reader who wants it,
                      never the only form: the label carries the claim. */}
                  <span
                    className={`pp-pop-news-tag${tagged ? ' is-exact' : ''}`}
                    title={attr.title}
                  >
                    {attr.label}
                  </span>
                </div>
              </div>
            </li>
          )
        })}
      </ul>
      {/* No pager for a player whose stories all fit on one page: two dead
          arrows under a list of three is a control that only ever says no.
          The range is stated rather than a page number -- "6-10" is where
          you are in the stories, which is what a reader is holding; "page 2
          of 4" is where you are in the pages, which is the card's own
          bookkeeping. The total stays in the head, so it is not said twice.

          Left is newer and right is older, the way the list itself runs.
          Both carry a word for it: the arrows are the affordance, the
          `aria-label` is the meaning, and neither is asked to do the other's
          job. */}
      {pages > 1 && (
        <div className="pp-pop-news-pager">
          <span className="mono pp-pop-news-range">
            {start + 1}&ndash;{start + shown.length}
          </span>
          <button
            type="button"
            className="pp-pop-news-page"
            onClick={() => setPage(at - 1)}
            disabled={at === 0}
            aria-label={at === 0 ? 'No newer stories' : 'Newer stories'}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
                 strokeLinejoin="round" aria-hidden="true">
              <path d="M15 5 8 12l7 7" />
            </svg>
          </button>
          <button
            type="button"
            className="pp-pop-news-page"
            onClick={() => setPage(at + 1)}
            disabled={at === pages - 1}
            aria-label={at === pages - 1 ? 'No older stories' : 'Older stories'}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
                 strokeLinejoin="round" aria-hidden="true">
              <path d="m9 5 7 7-7 7" />
            </svg>
          </button>
        </div>
      )}
    </PopCard>
  )
}
