# Auto-refit audit log

One row per run of `pipeline.auto_refit` that did real work (a fit + validation). Skips -- runs that found too little new data -- are logged to the daemon's stdout, not here. `nll` is log-loss (lower better); `t1`/`t5` are held-out top-1/top-5 (higher better). champion = the live artifact scored full-width; candidate = the refit scored leave-one-draft-out; both on the identical human picks. Decision is the gate's verdict.

| when (UTC) | corpus | labelled | champ t1 | cand t1 | champ nll | cand nll | flat t1 | decision | commit |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-24T15:47:47+00:00 | 204 | 181 | 0.2668 | 0.2668 | 2.6493 | 2.6493 | 0.2313 | no-swap | - |
| 2026-08-25T17:44:26+00:00 | 666 | 643 | 0.2693 | 0.2693 | 2.6217 | 2.6217 | 0.2341 | no-swap | - |
