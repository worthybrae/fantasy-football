// WHY THE DRAFT BUTTON IS OFF, IN THE WORDS OF THE THING THAT IS WRONG.
//
// Every disabled Draft button in the room used to say "Not your turn yet",
// which is one of three different situations and the least likely of them to
// need explaining. The clock can be on your seat and the button still dead:
// the send socket may be detached mid-reconnect (POST /api/live/select would
// answer 503), or the room may be unpaid. Both look identical from the
// button's side -- disabled -- and only one of them will fix itself.
//
// One function, because the two surfaces that draw the button in the simple
// view must not word it differently, and because "may he send this pick" is
// already computed once in DraftRoom and should not be re-derived here.

export interface DraftGate {
  /** The room's one gate: his turn, on a live socket, in a room he has paid
   *  for. True means the button is live and needs no explaining. */
  isMyTurn: boolean
  /** Whether the pick on the clock is his, regardless of the socket. */
  youAreUp: boolean
  /** `state.socket_alive` -- whether a pick could actually be sent. */
  socketAlive: boolean
  /** `billing.required` -- the room is readable but not draftable. */
  locked: boolean
}

/** The button's `title` while it is disabled, or `undefined` while it is not.
 *
 *  The socket is asked about FIRST: a detached handle is transient and comes
 *  back on its own, and telling somebody holding the clock that it is not
 *  their turn is the one wording that is flatly untrue. */
export function draftHint(gate: DraftGate): string | undefined {
  if (gate.isMyTurn) return undefined
  if (gate.youAreUp && !gate.socketAlive) return 'Reconnecting to ESPN…'
  if (gate.locked) return 'Unlock to draft'
  return 'Not your turn yet'
}
