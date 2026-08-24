# Auto-refit audit log

One row per run of `pipeline.auto_refit` that did real work (a fit + validation). Skips -- runs that found too little new data -- are logged to the daemon's stdout, not here. `nll` is log-loss (lower better); `t1`/`t5` are held-out top-1/top-5 (higher better). champion = the live artifact scored full-width; candidate = the refit scored leave-one-draft-out; both on the identical human picks. Decision is the gate's verdict.

| when (UTC) | corpus | labelled | champ t1 | cand t1 | champ nll | cand nll | flat t1 | decision | commit |
|---|---|---|---|---|---|---|---|---|---|
