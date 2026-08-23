import { Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing'
import DraftRoom from './pages/DraftRoom'
import MockDrafts from './pages/MockDrafts'
import './App.css'

// Two of these three routes are one path through the app: the landing page
// hands over the bookmarklet, clicking it in an ESPN draft room lands back
// here with a token and redirects into the draft room. A player on the
// board there opens as a popup overlay over the room (PlayerOverlay.tsx) --
// never its own route, the same way ESPN's own draft room does it -- so
// there is no route for a player profile. The standalone research board that used to live under
// /legacy is gone -- the draft room is the tool now, built around the
// ranked, need-adjusted recommendation rather than the old EV grid.
//
// /mocks is the one route off that path: the farm's own record of every ESPN
// mock draft it has joined, live or finished, with each board shaded by who
// actually made each pick. It is a reading room for drafts already played,
// not part of drafting one, so nothing in /draft links to it.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/draft" element={<DraftRoom />} />
      <Route path="/mocks" element={<MockDrafts />} />
    </Routes>
  )
}
