import { Suspense, lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing'
import MobileGate from './components/MobileGate'
import './App.css'

// LANDING IS THE ONLY ONE IN THE FIRST BUNDLE. Everybody arrives at `/`,
// and until this file was split everybody also downloaded the draft room,
// the archive, its data table, the league and manager pages, the mock
// listing and the report -- nine pages nobody on the landing page is
// looking at, in one chunk that had to parse before anything painted.
//
// Each `lazy` below is its own chunk, fetched when its route first matches.
// The cost is a network round trip on the way into a page; against that,
// every one of these pages is entered from a click that already involves a
// fetch of its own, and the chunk is fetched in parallel with it. The
// document, the shared vendor chunk and the landing page are what a visitor
// pays for now.
const DraftRoom = lazy(() => import('./pages/DraftRoom'))
const MockDrafts = lazy(() => import('./pages/MockDrafts'))
const Market = lazy(() => import('./pages/Market'))
const ArchiveData = lazy(() => import('./pages/ArchiveData'))
const LeaguePage = lazy(() => import('./pages/LeaguePage'))
const ManagerPage = lazy(() => import('./pages/ManagerPage'))
const Live = lazy(() => import('./pages/Live'))
const WaitingRoomPage = lazy(() => import('./pages/WaitingRoomPage'))
const LeagueReport = lazy(() => import('./pages/LeagueReport'))

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
    // One boundary around every route rather than one per page: the fallback
    // is a blank pane holding the page's height, and a nested boundary would
    // only let it appear in more places. It is deliberately empty -- a
    // spinner that shows for the 40ms a local chunk takes is worse than
    // nothing, and every page below paints its own loading state for the
    // data it then goes and fetches.
    <Suspense fallback={<div className="page-loading" />}>
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
                {/* One of your leagues, as a place: what it is, when it drafts, the
                    way in -- and where its report cards and profiles will live. */}
                <Route path="/league/:leagueId" element={<LeaguePage />} />
                {/* One manager in that league, over every season. */}
                <Route path="/league/:leagueId/manager/:memberId" element={<ManagerPage />} />
              </Routes>
            </MobileGate>
          }
        />
      </Routes>
    </Suspense>
  )
}
