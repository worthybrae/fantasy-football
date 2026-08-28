# Putting Cloudflare in front of espnfantasydraft.com

Written 2026-08-28, against the Cloudflare Free plan. Nothing here has been
done yet: the nameserver switch is the owner's to make, and this is the list
of what to click and what each click is for.

## What this buys

The app runs as one Railway container. Every visitor's document, bundle,
fonts, demo video and ADP page is served by that one process, and the
process is also the thing that has to answer a draft room polling twice a
second. Cloudflare in front of it takes the first group away entirely: the
hashed bundle and the fonts are marked `immutable` and get fetched from
Railway once ever, the ADP pages and the public API reads are marked
`s-maxage`, and what reaches the origin is the traffic that genuinely has to.

The headers that make this work are already in the app -- see
`api/http_cache.py`. Everything under `/api` that did not explicitly opt in
is `private, no-store`, so nothing with a reader's name on it can be cached
by anybody no matter what rule is configured here.

## Before you start

- The origin is `qoe90w4n.up.railway.app`. Railway terminates TLS there with
  its own certificate, which is why Full (strict) works below.
- DNS is at Namecheap today. The nameservers are `dns1.registrar-servers.com`
  and `dns2.registrar-servers.com`.
- `ESPN_CUSTODY_TRUST_FORWARDED_PROTO=1` is already set in the Dockerfile.
  That is what lets the credential guard see HTTPS through a proxy that
  forwards plain HTTP, and it is already doing that job for Railway's own
  edge. Adding Cloudflare makes the chain one hop longer and changes nothing
  about the switch.

## Steps

1. **Add the site.** Cloudflare dashboard, Add a site, `espnfantasydraft.com`,
   Free plan. Cloudflare scans the existing DNS and imports what it finds.

2. **Copy the two nameservers Cloudflare gives you.** They are a pair like
   `xxx.ns.cloudflare.com` and `yyy.ns.cloudflare.com`, and they are specific
   to this account -- do not copy them out of somebody else's guide.

3. **Set them at Namecheap.** Domain List, Manage, Nameservers, switch from
   Namecheap BasicDNS to Custom DNS, replace both
   `dns{1,2}.registrar-servers.com` entries with the Cloudflare pair, save.

4. **Wait for Active.** Cloudflare emails when the zone goes from Pending to
   Active. It is usually minutes and can be a few hours. The site stays up
   throughout: until the switch propagates, resolvers keep answering from
   Namecheap.

5. **SSL/TLS mode: Full (strict).** SSL/TLS, Overview, Full (strict). This
   means Cloudflare talks HTTPS to Railway and checks Railway's certificate,
   which Railway has and which is valid. Do not use Flexible: it would send
   plain HTTP to the origin, and the credential guard is entitled to refuse
   that.

6. **DNS records.** Two of them, both proxied (the orange cloud on, which is
   what actually puts Cloudflare in the path):

   | Type | Name | Target | Proxy |
   |---|---|---|---|
   | CNAME | `@` | `qoe90w4n.up.railway.app` | Proxied |
   | CNAME | `www` | `qoe90w4n.up.railway.app` | Proxied |

   Delete whatever the import left pointing at Railway, so there is one
   record per name and no stale A record underneath.

7. **One cache rule.** Caching, Cache Rules, Create rule.

   - Name: `Cache everything, respect origin`
   - When incoming requests match: `Hostname` `equals` `espnfantasydraft.com`
   - Then: Cache eligibility `Eligible for cache`
   - Edge TTL: `Use cache-control header if present, use default otherwise`
   - Browser TTL: `Respect origin TTL`

   The "respect origin" part is the whole point. Cache-everything on its own
   would cache HTML documents that must not be cached; respecting the origin
   means each response is held for exactly as long as the app said, and the
   app has an opinion about every one of them.

8. **Speed, Brotli on.** Speed, Optimization, Content Optimization, Brotli.
   The app gzips anything over a kilobyte itself; Brotli at the edge is
   better than gzip on the same bytes and costs nothing to switch on.

9. **Security, Bots on.** Security, Bots, Bot Fight Mode. Free-plan bot
   protection. It challenges obvious automation and leaves search crawlers
   alone, which matters because the ADP pages exist to be crawled.

10. **One rate limit, on the connects.** Security, WAF, Rate limiting rules,
    Create rule. Match: URI Path equals `/api/live/connect-token` OR URI
    Path equals `/api/live/connect`, request method POST. Rate: 10 requests
    per 1 minute, per IP. Action: Block for 10 seconds. A connect is the one
    request that costs the app real work -- a league file, a board, a
    socket -- and a browser retrying in a loop, or a script, must not be
    able to turn that into a queue everybody else waits behind. Ten a
    minute is far more than a person clicking a bookmark ever needs; the
    app's own room cap (`LIVE_MAX_ROOMS`) is the other half of this.

## The two streams

`/api/live/events` and `/api/demo/events` are server-sent event streams that
stay open for up to an hour. Neither needs a rule of its own: both send
`Cache-Control` saying not to store them (`no-cache` on the live one,
`no-store` on the demo one), the cache rule above respects that, and
Cloudflare passes streaming responses through rather than buffering them.

If a draft room ever does go quiet behind the proxy, the thing to check
first is whether some later rule made those paths cacheable -- not the
streams themselves.

## Checking it worked

From a machine that is not the origin:

```
curl -sI https://espnfantasydraft.com/ | grep -i 'cf-cache-status\|cache-control'
curl -sI https://espnfantasydraft.com/assets/<some-hashed-file>.js | grep -i 'cf-cache-status\|cache-control'
curl -sI https://espnfantasydraft.com/api/lobby | grep -i 'cf-cache-status\|cache-control'
```

Expected: the document says `no-cache` and is a `DYNAMIC` or `BYPASS` at the
edge, the asset says `public, max-age=31536000, immutable` and turns `HIT`
on the second request, and `/api/lobby` says `s-maxage=15` and turns `HIT`
within its window. Anything under `/api` that is not on the public list
should say `private, no-store` and never be a `HIT`.
