# Freeze manifest (Phase 1.5b, plan Task 3)

Written before any 2018+ inference. git HEAD at writing: `5dd0326a459d85d305f30746ea0abd8c61df68f5`. Experiment `R5_f3_alpha0_train`, arms HH and EH, seeds 0-4, 100 epochs, trained on Kaggle GPU with `save_weights=true`.
Selection rule: epoch with the best pre-2017 validation Sharpe (`best_epoch`, validation = 2016). No 2017 or 2018+ return affected selection.

## Runs

| arm | seed | best_epoch | epochs_run | val SR | 2017 test SR | sha256 config.json | sha256 best_state.pt | sha256 test_pred.npy |
|---|---|---|---|---|---|---|---|---|
| HH | 0 | 41 | 100 | 2.2336 | 2.1990 | `5b6714679509eb8913c9903f0df7b53d7c39cc87bdf1d900bc442b806f43905f` | `c0e0e987a19d94bf3cd1a8afe7adefb9e90bcb42ae7b76fa4be9c940dddf7fbd` | `e613c0d7f3b8149efbcd0ee731f662c425b87d5425862fcef205bf3af7ab24b1` |
| HH | 1 | 5 | 100 | 2.1155 | 2.8822 | `e46fbe1ff7a598164b213b5ffeff8f6c083d5dd36a62abf89fab52fac7e17490` | `d7335f33374440cc21dd81903df3a81ad8bf100a87ac43c59abaec9406705f24` | `10344a536930cc51ec3f18526de6075ce486c84ad039abbb55f9fa6a8736aee4` |
| HH | 2 | 0 | 100 | 2.7559 | 1.9592 | `6c14aeeb727c92e0491eea607039b185b19373a65578a57e3b749342666440b1` | `ef52e1422dc489d845ab87775252b4c8654aa6eb1b6e17fff4a41849f968cc1f` | `55afa9321bec3132060b31ff48bfab1d70e6f07eff0ed579a1ef567f8c5a9f08` |
| HH | 3 | 6 | 100 | 2.3117 | 1.8179 | `2d85be490de43eea5231b21000a2c6de8681d6b1ff545ffc630485e7d01a392b` | `75d16e38bcddc07f143b753c1d47b1b5f183c2a20c9963a1c5642ce2ca0ed1e8` | `06508e5e899876ab3d743c2b4df6a736df6b2a36e613859592bc8243e8a94cd9` |
| HH | 4 | 2 | 100 | 2.6670 | 2.1434 | `b9f77eba15a13570f8facd5ca6d3ab94d619bd42fe56843f91e7b32f659eb5b1` | `fb547ac9057fb1d0d0e31037a294722edeca3754d2bd0c6ac6f215fcea944282` | `4ec5ee34a52840d3b8c171eb8a4de37cca26d9b2776c8069a3f1deb628d95de0` |
| EH | 0 | 59 | 100 | 1.9894 | 1.4215 | `3dc10ebab6a532553d03940f08e86fd54efb0a5b55204a0d206f7758fbf57b2a` | `20684d148fc9a0813ee2e332e00aeb0db250e05176abcfb05df1fc08a1bcf73a` | `9818de22b2240523729f41ff0b209853c8306840eae340aafc507a073dead047` |
| EH | 1 | 97 | 100 | 1.9831 | 0.5244 | `0ed4c10e8e0dd75a88641c6deff782049af4888fc2ccea00a1bd9e71250f5ffa` | `0e78f54e4f587989a9e64f59febaab4b98c3a6c467e0b193c1dc8d30dbc50f69` | `7ee5b76605a5a5fe4dd80394ccf1d68d5336f1cd20039823f81071a146e2e01a` |
| EH | 2 | 21 | 100 | 2.3694 | 1.5785 | `b472919d108a000c0b82d85f38a429a32044c09d7ea4d801c9fffcc816ac6918` | `7ca027da64be0687c1b4d88da215ab0c39fe73cd62d003d964a768f4f5d3453d` | `d9c560535c229484d099b77b1371b76abfe9b87a6590cd24826f30d4f3b7ae09` |
| EH | 3 | 19 | 100 | 1.8265 | 2.4604 | `3dd50105537b20f6e9485b7352c1ebaf537762597e4d50e2887f9ffb62c42d5f` | `798035b138886c7f3df77505c918dbecc96c20d17fbb8cb725092073b2cdd4ed` | `944c2dc241a4b383d0cb66582535b9ec4b1d3aa89ded9e537a8a84db5cd94937` |
| EH | 4 | 1 | 100 | 1.9128 | 1.3321 | `7c7a4fadd8e997a1006f383c5461fefc150a869d871ebd2bbf66874782c2fa33` | `b4ac6d98f94264f619075d34ede7425da5c13e2575a8ea99e1daf969771cc58c` | `e1c36bbbbf9fe9953a6537234c89c1482e801c585f9af1a5cd606ccd878e6ff0` |

## Graph and data inputs (sha256)

| input | path | sha256 |
|---|---|---|
| graph v2 cache (hypergraph used by all runs) | `data/raw/rsr/data/hypergraph_cache/NYSE_industry-wiki.json` | `859d44173f0dd61c896d63a22af84455fecae4a3d66461a118cf80ce802e197c` |
| RSR ticker order | `data/raw/rsr/data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv` | `fb14b5ef35894a1d7d8a80f3ba1c750c0a628c6e3d8d2c3e75dad2fc0dc11701` |
| RSR calendar | `data/raw/rsr/data/NYSE_aver_line_dates.csv` | `2a883ed7238a42057b9259a85c9319f3a75699c39c31477559c6ac51b9aa3796` |
| Alpaca raw bars (all adjustments) | `data/raw/alpaca_post2017/bars.pkl.gz` | `b0609701578e17ff02e42137ee307a189a34b0de70c23015528458c1a1722bea` |
| Alpaca download meta | `data/raw/alpaca_post2017/download_meta.json` | `af4ad1ea3068a8e29b5a6fabc53b2307efc3ab30487edb3c4f8559d219169487` |
| Alpaca identity-pass list (per-ticker audit table) | `docs/phase1_5b/alpaca_audit_tickers.csv` | `33a4f05b6a813d79b4ae33f894118654c03cb38405bfeb8250685a54b0dbe841` |
| Alpaca audit results | `docs/phase1_5b/alpaca_audit_results.json` | `7373afad5793b522423d1b490a0c411caf36304d0bc756c375cf1c8dbcdcbe61` |

## R7: 2017 replication check (predeclared, plan R7)

Rule: pass if the new mean HH test Sharpe over seeds 0-4 lies in [min - SD, max + SD] of the historical `R5_f2_alpha0_train/HH` seeds (SD = sample SD, ddof=1, of the historical seeds). Inference proceeds either way.

- Historical HH seeds: [1.881, 1.931, 1.895, 2.035, 2.143]; mean 1.977, min 1.881, max 2.143, SD 0.111. Interval [1.771, 2.254].
- New HH seeds: [2.199, 2.882, 1.959, 1.818, 2.143]; mean **2.200**.
- **R7 PASS** (new HH mean 2.200 inside [1.771, 2.254]).
- EH (descriptive only): historical [3.003, 2.106, 1.447, 2.444, 1.247] mean 2.049, SD 0.720; new [1.422, 0.524, 1.579, 2.46, 1.332] mean 1.463, SD 0.691.
- Also descriptive: HH-EH 2017 mean difference, new +0.737, historical -0.072.

Note: this is a replication of the recipe, not of the historical weights (those do not exist); GPU nondeterminism makes the runs differ.


## Addendum A (2018-2023 locked inference, 2026-10-04)

One locked pass by `scripts/eval_post2017.py run` (git HEAD of the freeze commit `00cf6e4`, script added after). Panel: Alpaca adjustment=split, 1,647 identity-pass nodes, masked after last bar, 2018-01-02..2023-12-29 (1,509 target days), frozen graph v2 (4,350 edges). Outputs in `results/post2017_frozen/` (git-ignored). Pipeline check on 2017 (seed 0 HH, Alpaca panel vs stored RSR-panel predictions): SR 1.809 vs 2.199, mean daily cross-sectional Pearson 0.949.

Gross top-5 Sharpe per run: HH/0 0.8546, HH/1 0.0962, HH/2 0.7981, HH/3 0.2265, HH/4 0.1935, EH/0 0.2669, EH/1 0.1436, EH/2 0.2037, EH/3 0.1128, EH/4 0.0754.

sha256 of outputs (full list in `results/post2017_frozen/OUTPUT_HASHES.json`, sha256 of that file below):

| file | sha256 |
|---|---|
| `EH/seed_0/test_daily.npy` | `946320ef93e03804478a2a20e63f29538fd5f6d81d5b8cc6c21f1bdbd78df06f` |
| `EH/seed_0/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `EH/seed_0/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `EH/seed_0/test_pred.npy` | `fddca319b0bef91103744d3c7f7cb14c5f0314ee8ede00470180531090dc5bad` |
| `EH/seed_1/test_daily.npy` | `aaee147fc996299d99714e504081cf650048dbef8b3c036afd36efea6b08e66b` |
| `EH/seed_1/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `EH/seed_1/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `EH/seed_1/test_pred.npy` | `15d99ac55be6716bb3081a96ab7930f4550f776e8806b74f4f078beee01fca61` |
| `EH/seed_2/test_daily.npy` | `2648e4f66bf5a172e8752596407bdc5fe525a6ec789804a03a348d07522f5fc8` |
| `EH/seed_2/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `EH/seed_2/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `EH/seed_2/test_pred.npy` | `d2634c3498b03247bc7c14309dd90657f16921460b802d8a23972087148fefb8` |
| `EH/seed_3/test_daily.npy` | `71297f61dd1e5128c88c0f0e6b9f241c1210d5158a47ff8cd0aef05605b28b17` |
| `EH/seed_3/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `EH/seed_3/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `EH/seed_3/test_pred.npy` | `d262b56fb30530dd8d1a8806d3a4772442e6dae03f37236f9e0860aa8b25c00d` |
| `EH/seed_4/test_daily.npy` | `1f3721231b9e7a6567c03f5abb44ac8a5936e0586152527b48521f2b48011582` |
| `EH/seed_4/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `EH/seed_4/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `EH/seed_4/test_pred.npy` | `36d146737ca9688c9566369ae73cd97efeef362c5cae5f54cbe68295ade61946` |
| `HH/seed_0/test_daily.npy` | `a663d299e380db8e6b0c81c459714257797d547991314d607c114ad16b6c758b` |
| `HH/seed_0/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `HH/seed_0/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `HH/seed_0/test_pred.npy` | `a8854cd7e48096a957e276ce6ea482ead122087d070f626c76228ac1166604c5` |
| `HH/seed_1/test_daily.npy` | `6dc0837dc9d38cdb6bb6906d7d4d95c4dac669f745e2a28ba6d9510fdc6d848d` |
| `HH/seed_1/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `HH/seed_1/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `HH/seed_1/test_pred.npy` | `adcb50833c5e0d06ad14d9b4b30907bed63bba7126558cde8d57edd91f6cd1b2` |
| `HH/seed_2/test_daily.npy` | `80bf2d7ab6b23a526289c37a578be6f024ae53bfe541a482d6c7e0012ba29e66` |
| `HH/seed_2/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `HH/seed_2/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `HH/seed_2/test_pred.npy` | `cc99414c974b1c14319e0d03e6ebda062c8ee3eab2e6ac849c6421aad431ae69` |
| `HH/seed_3/test_daily.npy` | `5600238b47eece5ff6febf59b82a508e9551cada59393708c605ba432679eae0` |
| `HH/seed_3/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `HH/seed_3/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `HH/seed_3/test_pred.npy` | `e41eed56f5895eea81cca35e4be4d019f5e81ab347484d90e89c4d6a36846769` |
| `HH/seed_4/test_daily.npy` | `c41bff99e89b0c1ae89ccbee17d950b79794c66a26614719a681239b9192c95b` |
| `HH/seed_4/test_gt.npy` | `afbc392426a6f7cf705c4a0d46fceadfcef6fa2b160d31366da2fa0a73a13288` |
| `HH/seed_4/test_mask.npy` | `46a1beecaa6e5b54774f2b0d4985d6ddbd2afa764d598cbb899b3bfd9c8df3d4` |
| `HH/seed_4/test_pred.npy` | `778e18aa48cecf804c3c4f3f0c48aa4844da90be54db635f4540b560f956a90b` |
| `dates.npy` | `e4a463aeb7acc3eeb3d4d09aa4b38f60c461ce6d94e1befce03fcd13a174178a` |
| `OUTPUT_HASHES.json` | `565f9927c13dd0dd5338d41d8735aef0fa82df392637c085b25285d54fa8139e` |
