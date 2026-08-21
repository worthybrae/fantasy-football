export interface VerdictFigure {
  label: string
  value: string
  accent?: boolean
}

// The verdict, first: the handful of numbers the pick is actually made on,
// in one strip across the top of the card. Everything below it is the
// argument for or against that verdict, which is the whole shape of this
// design -- lead with the answer, then show the three things ESPN
// structurally cannot.
//
// One strip, two possible sources, and they are never mixed. In the draft
// room the figures come from the seed the room already held (gain vs
// waiting, survival, which roster slot he fills -- numbers the profile
// endpoint does not have and could not compute, since they are undefined
// outside a draft), painted on the frame the overlay opens. With no seed at
// all (PlayerProfile's un-embedded fallback, which nothing currently opens
// -- the /players/:slug route it used to back is gone) there is no seed and
// they are read off the board row in the response instead. Pre-formatted
// strings either way: the caller owns rounding and sign rules, this owns the
// geometry.
export default function VerdictStrip({ figures }: { figures: VerdictFigure[] }) {
  if (figures.length === 0) return null
  return (
    <div className="pp-verdict">
      {figures.map((f) => (
        <div key={f.label} className={`pp-verdict-cell${f.accent ? ' is-open' : ''}`}>
          <div className="pp-cap">{f.label}</div>
          <div className="pp-verdict-value mono">{f.value}</div>
        </div>
      ))}
    </div>
  )
}
