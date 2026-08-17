import { Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing'
import LiveDraft from './pages/LiveDraft'
import PlayerPage from './components/PlayerPage'
import './App.css'

// Three routes, one path through them: the landing page hands over the
// bookmarklet, clicking it in an ESPN draft room lands back here with a token
// and redirects to the live board, and a player on that board opens their
// profile. The standalone research board that used to live under /legacy is
// gone -- the live board is the tool now.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/draft" element={<LiveDraft />} />
      <Route path="/players/:slug" element={<PlayerPage />} />
    </Routes>
  )
}
