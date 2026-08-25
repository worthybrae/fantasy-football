// web/src/components/Explainer.tsx
import { useEffect } from 'react'

// The page in words, for the reader who wants them and the crawler that
// needs them. The first screen is a live draft and the second is drawn
// figures; neither says in plain text what this is. This does, once, below
// both, and it is the only <h1> on the page.
//
// The FAQ is real <details> elements and also FAQPage structured data, so
// the same six answers show under the result in a search as well as here.

const FAQ: { q: string; a: string }[] = [
  {
    q: 'Does it work with my ESPN league?',
    a: 'Yes, any ESPN fantasy football league whose draft room you can open in a browser — public or private, snake or auction-free formats, any roster settings. It reads your league’s scoring and roster slots and ranks for those.',
  },
  {
    q: 'Is this allowed?',
    a: 'It is a second browser tab beside your draft. It never picks for you, never touches the draft room, and never logs in as you anywhere. ESPN’s own draft tools do the same job with worse numbers.',
  },
  {
    q: 'What does the bookmark do?',
    a: 'One click on any ESPN fantasy page hands this site the draft session ESPN already gave your browser, so the board here follows every pick in your room live. Nothing is installed.',
  },
  {
    q: 'Does it see my ESPN password?',
    a: 'No. It never sees a password. It holds the same session cookie ESPN gave your browser, encrypted, and only for as long as your draft needs it. You can revoke it from the dashboard at any time.',
  },
  {
    q: 'What does it cost?',
    a: 'Mock drafts are free, as many as you like. A real draft is $9.99, once, paid when you connect the league that is about to draft.',
  },
  {
    q: 'Where do the rankings come from?',
    a: 'From hundreds of real ESPN mock drafts this site records every day, plus consensus ADP and projections. It knows who actually goes at pick 41 in an ESPN room, not just who a list says should.',
  },
]

export default function Explainer({ onStart }: { onStart: () => void }) {
  useEffect(() => {
    const script = document.createElement('script')
    script.type = 'application/ld+json'
    script.text = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'FAQPage',
      mainEntity: FAQ.map(({ q, a }) => ({
        '@type': 'Question',
        name: q,
        acceptedAnswer: { '@type': 'Answer', text: a },
      })),
    })
    document.head.appendChild(script)
    return () => { script.remove() }
  }, [])

  return (
    <section className="xp" aria-labelledby="xp-h1">
      {/* TWO COLUMNS AT THE PAGE'S OWN WIDTH. The prose wants a readable
          measure, which is why this block used to be a 720px column -- but a
          720px column inside a 1240px page reads as a stray, indented under
          everything above it. Splitting the words from the questions keeps
          both columns near 60 characters and lets the section start and end
          on the same gutter as the footer. One column again under 900px. */}
      <div className="xp-say">
        <h1 id="xp-h1" className="xp-h1">
          A draft assistant for ESPN fantasy football, trained on real ESPN mock drafts.
        </h1>
        <div className="xp-body">
          <p>
            ESPN Draft Assist is a live board that sits beside your ESPN draft room.
            Every pick in your room lands here as it happens, and the board re-ranks
            around it: who is worth taking now, who will still be there at your next
            turn, and who will not.
          </p>
          <p>
            It is ranked from what ESPN rooms actually do. This site records hundreds
            of real ESPN mock drafts every day, so when it says a player usually goes
            at pick 41, that is where he has been going this week, in rooms like yours.
          </p>
          <p>
            Setup is a bookmark. Drag it to your bar, click it once on ESPN, and your
            leagues appear here with a button to open the board when your draft starts.
            Mock drafts are free; a real draft is $9.99, once.
          </p>
        </div>
      </div>

      <div className="xp-ask">
        <h2 className="xp-h2">Questions</h2>
        <div className="xp-faq">
          {FAQ.map(({ q, a }) => (
            <details key={q} className="xp-q">
              <summary>{q}</summary>
              <p>{a}</p>
            </details>
          ))}
        </div>
      </div>

      <button type="button" className="lp-cta xp-cta" onClick={onStart}>
        Try a mock draft
      </button>
    </section>
  )
}
