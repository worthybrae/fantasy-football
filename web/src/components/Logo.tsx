// The Draft Assistant mark: a football on the diagonal.
//
// Kept as a component rather than an <img src="/favicon.svg"> so it inherits
// the surrounding colour and never flashes in after the text it sits beside.
// The same geometry ships as web/public/favicon.svg -- keep the two in step.
//
// The laces are drawn in the page canvas colour rather than stroked around the
// shell: at the 14-16px this renders at, an outline fills in and the mark reads
// as a plain amber blob.

export function Logo({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      style={{ flexShrink: 0 }}
    >
      <path fill="var(--accent)" d="M20 4A13 13 0 0 1 4 20 13 13 0 0 1 20 4Z" />
      <g stroke="var(--bg-0)" strokeWidth="1.5" strokeLinecap="round">
        <path d="M14.6 9.4 9.4 14.6" />
        <path d="M12.8 8.8 15.2 11.2" />
        <path d="M10.8 10.8 13.2 13.2" />
        <path d="M8.8 12.8 11.2 15.2" />
      </g>
    </svg>
  )
}
