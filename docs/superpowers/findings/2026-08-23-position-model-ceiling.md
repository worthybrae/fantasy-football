# Position-prediction ceiling

**Diagnostic. Held out by draft, human picks only.**

```
drafts 66, human picks 4625

always_wr        acc 0.3743 +/-0.0035   early 0.339 mid 0.488 late 0.298
majority_round   acc 0.4649 +/-0.0096   early 0.592 mid 0.492 late 0.363
round_roster     acc 0.5029 +/-0.0073   early 0.589 mid 0.527 late 0.429
multinomial      acc 0.5304 +/-0.0085   early 0.587 mid 0.557 late 0.473
mlp              acc 0.5170 +/-0.0097   early 0.638 mid 0.528 late 0.432

best position acc 0.5304  x within-pos 0.55 => implied nested top-1 ceiling ~ 0.2917
(champion flat model today: 0.2595)
```
