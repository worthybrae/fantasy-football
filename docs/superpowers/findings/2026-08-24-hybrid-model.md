# Hybrid opponent model (flat early, nested mid/late)

**Diagnostic. Held out by draft, human picks only. Routes per pick: flat prior in the early bucket, nested from mid on.**

```
drafts 84, human picks 6107

champion   top1 0.2306 +/-0.0066  top3 0.4760  top5 0.6190  logloss -2.9000
nested     top1 0.2512 +/-0.0074  top3 0.5363  top5 0.6836  logloss -2.6474
HYBRID     top1 0.2605 +/-0.0072  top3 0.5379  top5 0.6856  logloss -2.6911

hybrid - champion top1: +0.0300
hybrid - nested   top1: +0.0093

by round bucket (hybrid uses flat early, nested mid/late):
  early  n= 1496  champ 0.3783  nested 0.3402  hybrid 0.3783
  mid    n= 2120  champ 0.2259  nested 0.2392  hybrid 0.2392
  late   n= 2491  champ 0.1457  nested 0.2079  hybrid 0.2079

best overall top-1: HYBRID (0.2605)
```
