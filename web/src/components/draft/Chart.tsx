// Lives on its own because the player profile popup is about to draw the
// same picture the board's hover panels draw -- same columns, same baseline,
// same colour rules. A second `Chart` written for the popup is exactly how
// the two would drift: one gets a fix or a new column type and the other
// quietly doesn't.
import type { ReactNode } from 'react'

// -- one chart, drawn three times ------------------------------------------
//
// Every panel is the same picture: time along the bottom, one column per
// period, something growing from a shared baseline, the value above it and
// the period below. Health and Finish are both about seasons and used to be
// read in opposite directions -- one a stack of rows, the other a left-to-
// right axis -- so a reader had to learn the panel again on each column.
//
// Taller is better in all three and the colour means the same thing in all
// three, which leaves one thing to know per panel: what a column is.
export type Col = {
  key: string | number
  label: string                           // under the baseline: a year, a team
  value: string                           // above the bar
  tone: string                            // the shared five-step colour
  fill: number                            // 0..1 of the slot
  units?: { filled: number; of: number }  // circles instead of a solid bar
  empty?: boolean                         // did not play: a baseline mark
  projected?: boolean                     // has not happened: drawn hollow
}

export function Chart({ cols }: { cols: Col[] }): ReactNode {
  return (
    <div className="ctip-chart">
      {cols.map((c) => (
        <span key={c.key} className={`ctip-col${c.projected ? ' is-forecast' : ''}`}>
          <span className={`ctip-col-val ${c.empty ? 'is-off' : c.tone}`}>
            {c.value}
          </span>
          <span className="ctip-col-slot">
            {c.empty
              // Not a zero-height bar: "did not play" and "played and scored
              // nothing" are different claims, and the second already draws
              // as the 6% floor below.
              ? <span className="ctip-col-none" />
              : c.units
                ? <span className="ctip-units">
                    {Array.from({ length: c.units.of }, (_, i) => (
                      <span key={i} className={`ctip-unit${
                        i < (c.units?.filled ?? 0) ? ` is-on ${c.tone}` : ''}`} />
                    ))}
                  </span>
                : <span className={`ctip-col-bar ${c.tone}${c.projected ? ' is-proj' : ''}`}
                        style={{ height: `${Math.max(6, c.fill * 100)}%` }} />}
          </span>
          <span className="ctip-col-label">{c.label}</span>
        </span>
      ))}
    </div>
  )
}
