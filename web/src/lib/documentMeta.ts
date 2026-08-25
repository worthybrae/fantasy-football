import { useEffect } from 'react'

// One hook, one place, for what a route says about itself to a crawler
// and a browser tab: the title, the description, the canonical URL, and
// whether the page is a page at all. The draft room and the live board are
// application surfaces -- a search result that landed somebody inside a
// draft with no draft would be worse than no result -- so they ask not to
// be indexed, and everything else asks to be.
//
// `web/index.html` carries the landing page's static head for the crawler
// that never runs JavaScript. This hook is what the SPA does once it has.

const ORIGIN = 'https://espnfantasydraft.com'

function upsert(selector: string, create: () => HTMLElement): HTMLElement {
  const found = document.head.querySelector<HTMLElement>(selector)
  if (found) return found
  const made = create()
  document.head.appendChild(made)
  return made
}

function meta(name: string): HTMLMetaElement {
  return upsert(`meta[name="${name}"]`, () => {
    const el = document.createElement('meta')
    el.setAttribute('name', name)
    return el
  }) as HTMLMetaElement
}

export function useDocumentMeta({ title, description, canonical, noindex = false }: {
  title: string
  description?: string
  canonical?: string
  noindex?: boolean
}): void {
  useEffect(() => {
    document.title = title
    if (description !== undefined) meta('description').setAttribute('content', description)
    const link = upsert('link[rel="canonical"]', () => {
      const el = document.createElement('link')
      el.setAttribute('rel', 'canonical')
      return el
    })
    link.setAttribute('href', canonical ?? ORIGIN + window.location.pathname)
    const robots = document.head.querySelector('meta[name="robots"]')
    if (noindex) {
      meta('robots').setAttribute('content', 'noindex')
    } else if (robots) {
      robots.remove()
    }
  }, [title, description, canonical, noindex])
}
