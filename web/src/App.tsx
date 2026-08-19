import { Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing'
import DraftRoom from './pages/DraftRoom'
import PlayerPage from './components/PlayerPage'
import './App.css'

// Three routes, one path through them: the landing page hands over the
// bookmarklet, clicking it in an ESPN draft room lands back here with a token
// and redirects into the draft room, and a player on the board there opens
// their profile. The standalone research board that used to live under
// /legacy is gone -- the draft room is the tool now, built around the
// ranked, need-adjusted recommendation rather than the old EV grid.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/draft" element={<DraftRoom />} />
      <Route path="/players/:slug" element={<PlayerPage />} />
    </Routes>
  )
}
