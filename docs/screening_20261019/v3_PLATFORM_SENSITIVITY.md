# v3 platform sensitivity

Exploratory. 51 v3 fits ran on the laptop CPU (torch 2.14), including the common parent `static`. 120 ran on Kaggle CPU (torch 2.11). To measure the platform effect, `static` was rerun on Kaggle with the same source hashes, data, seeds 7/19/42 and folds 2022/2023, into `results/SCREEN_SP500_20261019_v3_platform_static` (kernel `hypershift-screen-v3-pc`).

## Per-run comparison (Kaggle static − laptop static)

| Fold | Seed | RankIC Kaggle | RankIC laptop | Diff | Prediction corr |
|---|---|---|---|---|---|
| 2022 | 7 | +0.00111 | +0.00134 | −0.00023 | 0.99987 |
| 2022 | 19 | −0.01271 | −0.01308 | +0.00037 | 0.99884 |
| 2022 | 42 | −0.00938 | −0.00940 | +0.00002 | 1.00000 |
| 2023 | 7 | +0.00016 | +0.00019 | −0.00004 | 1.00000 |
| 2023 | 19 | +0.01614 | +0.02366 | −0.00752 | 0.97352 |
| 2023 | 42 | +0.03821 | +0.03997 | −0.00176 | 0.95311 |

Four of six runs agree to within 0.0004 RankIC. Two 2023 runs diverge (prediction correlation 0.95–0.97). Small numeric differences sent them down different training paths, which is run-to-run jitter of seed-noise size, not a systematic bias. Mean −0.0015 (2022 +0.0001, 2023 −0.0031).

## Re-pairing the Kaggle-run arms against Kaggle static (pooled ΔRankIC [95% CI], gate label)

```
arm                               vs laptop static                   vs kaggle static
residual_soft   +0.0018 [-0.0142,+0.0187] Neutral    +0.0033 [-0.0117,+0.0194] Promising
sector_target   -0.0240 [-0.0569,+0.0086] Negative p -0.0225 [-0.0545,+0.0098] Negative p
hybrid          +0.0072 [-0.0032,+0.0181] Neutral    +0.0087 [-0.0016,+0.0193] Promising
multiscale      +0.0017 [-0.0113,+0.0154] Neutral    +0.0032 [-0.0093,+0.0168] Neutral
summary         +0.0013 [-0.0169,+0.0214] Neutral    +0.0028 [-0.0157,+0.0229] Neutral
adaptive        -0.0017 [-0.0257,+0.0234] Neutral    -0.0002 [-0.0247,+0.0257] Neutral
soft            +0.0049 [-0.0113,+0.0206] Neutral    +0.0064 [-0.0098,+0.0225] Neutral
random_graph    +0.0030 [-0.0055,+0.0113] Neutral    +0.0045 [-0.0034,+0.0124] Promising
sector_graph    +0.0038 [-0.0058,+0.0129] Neutral    +0.0053 [-0.0048,+0.0147] Promising
market_target   -0.0264 [-0.0591,+0.0061] Negative p -0.0249 [-0.0568,+0.0072] Negative p
cluster_target  -0.0172 [-0.0457,+0.0112] Negative p -0.0157 [-0.0432,+0.0121] Negative p
listwise        +0.0059 [-0.0161,+0.0281] Neutral    +0.0074 [-0.0159,+0.0310] Neutral
pairwise        -0.0046 [-0.0120,+0.0020] Negative p -0.0030 [-0.0091,+0.0023] Negative p
lowrank         -0.0056 [-0.0170,+0.0055] Negative p -0.0040 [-0.0148,+0.0063] Negative p
mix             -0.0080 [-0.0365,+0.0210] Negative p -0.0065 [-0.0351,+0.0225] Negative p
none            +0.0042 [-0.0111,+0.0195] Neutral    +0.0058 [-0.0097,+0.0214] Neutral
mlp             +0.0040 [-0.0153,+0.0228] Neutral    +0.0055 [-0.0139,+0.0242] Neutral
```

**Reading.** Re-pairing shifts every Kaggle arm by the same ≈ +0.0015, so it says nothing new about any single arm. It does flip four gate labels to 'Promising', including the negative control `random_graph`. The 3-seed 'Promising' gate therefore cannot separate arms within about 0.002 of its 0.003 threshold. **The predeclared v3 labels (laptop-static pairing) remain primary.** Every CI includes 0 under both pairings, and no Holm conclusion changes.
