import type { ReactNode } from 'react'
import CardHint from './CardHint'

// The popup's one card, and the only one: a title, the yardstick the numbers
// under it are read against, and the card's own body.
//
// The four season panels above these rows wear the same frame -- PopPanel
// passes `.pp-pop-panel` through `className` for the rules that size a chart
// inside it -- so a dense card in the lower rows and a chart panel in the
// upper one are the same object at the same size, and there is one place to
// change what a card in this popup looks like.
export default function PopCard({ title, note, hint, className = '', children }: {
  title: string
  // Omitted, not dashed, when the card has no yardstick to state: an empty
  // note would reserve the space and say nothing in it.
  note?: ReactNode
  // What this card measures, on hover of its own title (CardHint). Every
  // card in the popup passes one -- the copy lives in `hints.ts` -- and the
  // prop stays optional so a card built elsewhere is not forced to invent an
  // explanation it does not have.
  hint?: string
  className?: string
  children: ReactNode
}) {
  return (
    <section className={`pp-pop-card${className ? ` ${className}` : ''}`}>
      {/* The hover panel's own head (see App.css), which the season panels
          already use: two cards side by side cannot label themselves two
          different ways. The title is a real heading -- every card in the
          popup comes through here, so this is the one place that decides
          whether the popup is a document a screen reader can walk or a
          picture of one. `.pp-pop-card .ctip-head > h3` takes the browser's
          own heading styling back off it. */}
      <div className="ctip-head">
        {/* The title carries the explanation, not a separate icon: fourteen
            question marks in one popup is fourteen new things on screen to
            explain one thing each. The dotted underline CardHint draws is the
            whole affordance. */}
        <h3>{hint ? <CardHint text={hint}>{title}</CardHint> : title}</h3>
        {note !== null && note !== undefined && <span className="ctip-head-note">{note}</span>}
      </div>
      {children}
    </section>
  )
}

/** One label/value line. `tone` is emphasis, never data identity: `strong`
 *  is the card's own measurement, `accent` is the board's own number -- the
 *  same accent his board rank wears in the header. */
export interface PopRow {
  label: string
  value: ReactNode
  tone?: 'strong' | 'accent'
}

// The label/value stack three of the six cards are made of (Market, Usage,
// O-line). Stated once so a row in one card lines up with a row in the
// next -- they sit shoulder to shoulder and the eye reads across them.
export function PopRows({ rows }: { rows: PopRow[] }) {
  return (
    <div className="pp-pop-kv">
      {rows.map((r) => (
        <div className="pp-pop-kv-row" key={r.label}>
          <span className="pp-pop-kv-label">{r.label}</span>
          <span className={`mono pp-pop-kv-value${r.tone ? ` is-${r.tone}` : ''}`}>
            {r.value}
          </span>
        </div>
      ))}
    </div>
  )
}
