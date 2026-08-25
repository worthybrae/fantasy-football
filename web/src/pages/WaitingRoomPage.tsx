import { useCallback } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import type { TokenConnectParams } from '../api'
import WaitingRoom from '../components/WaitingRoom'
import { useDocumentMeta } from '../lib/documentMeta'

// The waiting room as a ROUTE: /room/:leagueId(?season=YYYY).
//
// It used to be a view state inside the landing page, which meant the URL
// never changed: a refresh dropped the reader back on the dashboard, the
// back button left the app, and a room could not be linked to. The page owns
// none of the room's logic -- that stays in components/WaitingRoom.tsx --
// it only binds the URL to the component's two exits:
//
//   back -> the dashboard, at /.
//   join -> also /, with the minted token in the HASH. That is the exact
//           shape the bookmarklet delivers (Landing's tokenFromHash), so the
//           connect screen, the retry and the board stay one path with one
//           more door into it. Landing wipes the hash on arrival, same as it
//           does for the bookmarklet's.
export default function WaitingRoomPage() {
  useDocumentMeta({ title: 'ESPN Draft Assist', noindex: true })
  const navigate = useNavigate()
  const { leagueId } = useParams()
  const [search] = useSearchParams()
  const season = search.get('season')

  const onJoin = useCallback((params: TokenConnectParams) => {
    const q = new URLSearchParams({
      leagueId: params.leagueId,
      teamId: params.teamId,
      swid: params.swid,
      token: params.token,
    })
    if (params.season) q.set('season', params.season)
    navigate({ pathname: '/', hash: q.toString() })
  }, [navigate])

  if (!leagueId) {
    navigate('/')
    return null
  }

  return (
    <WaitingRoom
      leagueId={leagueId}
      season={season === null ? null : Number(season)}
      onJoin={onJoin}
    />
  )
}
