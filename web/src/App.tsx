import { Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing'
import DraftRoom from './pages/DraftRoom'
import MockDrafts from './pages/MockDrafts'
import Market from './pages/Market'
import ArchiveData from './pages/ArchiveData'
import Live from './pages/Live'
import WaitingRoomPage from './pages/WaitingRoomPage'
import LeagueReport from './pages/LeagueReport'
import MobileGate from './components/MobileGate'
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
//
// Every one of them is desktop only -- the tool lives beside ESPN's draft room
// in a desktop browser -- so on a phone MobileGate replaces all of them with
// one page that says so and shows the room.
export default function App() {
  return (
    <Routes>
      {/* The league report is a link dropped in a group chat and opened on
          a phone. It is the one page that lives outside the desktop gate. */}
      <Route path="/leagues/:leagueId/report/:season" element={<LeagueReport />} />
      <Route
        path="*"
        element={
          <MobileGate>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/draft" element={<DraftRoom />} />
              <Route path="/mocks" element={<MockDrafts />} />
              {/* The draft archive: what hundreds of recorded drafts do from a given
                  seat. Signed in only -- the page itself offers the way in when the
                  API answers 403. */}
              <Route path="/archive" element={<Market />} />
              {/* The rows the archive is counted from, as a table. Same gate. */}
              <Route path="/archive/data" element={<ArchiveData />} />
              {/* Watch a live mock draft, from any page's tab strip. */}
              <Route path="/live" element={<Live />} />
              {/* A mock room before it starts: its seats, its countdown, the seat
                  you take. A route so the URL names the room and refresh keeps it;
                  joining navigates to / with the token in the hash, the same door
                  the bookmarklet uses. */}
              <Route path="/room/:leagueId" element={<WaitingRoomPage />} />
            </Routes>
          </MobileGate>
        }
      />
    </Routes>
  )
}
