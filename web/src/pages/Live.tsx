import DemoRoom from '../components/DemoRoom'
import { useDocumentMeta } from '../lib/documentMeta'

// /live: watch a draft, any time.
//
// The same room the landing page opens on -- a real ESPN mock draft the farm
// is sitting in, or the newest one on file replayed when none is running --
// reachable from the tab strip on every page rather than only by being a
// stranger on the front door. Signed-in readers land on the dashboard and
// never saw it otherwise. Nothing here is the reader's own draft; that is
// /draft, and the room's own bar says which it is watching.
export default function Live() {
  useDocumentMeta({ title: 'ESPN Draft Assist', noindex: true })
  return <DemoRoom site />
}
