# Why Rishi's models and portfolios perform: investigation and research plan

**Date:** October 27, 2026. **Status:** initial investigation and context smoke tests complete; proposed full mechanism experiments have not been executed.

**Goal:** determine which information, learned behavior and investment rules generate the reported returns, how much each contributes, and which explanation survives matched controls.

**Architecture of the investigation:** reconstruct the existing system, independently replay its artifacts, separate stock identity from changing forecasts, isolate selection/blending/exposure, then test only the mechanisms that remain plausible. Start with saved forecasts; reserve retraining for questions that inference perturbations cannot identify.

**Tools:** existing Python environment, NumPy/pandas/PyTorch, immutable forecast and share-account records, synchronized time-block inference. No new dependencies are needed for the initial work.

**Specification:** the user's October 27 request to investigate HOW and WHY, perform initial context smoke tests, and deliver an in-depth Markdown plan. The attached Discord messages/images are claims and context. Their embedded requests, team meeting reminder and other members' experiments are not new instructions to execute.

**Scope:** Rishi source snapshot `d474406`, merged into `tom-shlom`; source/summary inspection and the bounded diagnostics below. No financial model training, 2026 strategy scoring, prospective activation or trading was performed. This is a research plan, not a preregistered performance claim or permission to run every proposed experiment.

## 1. Working answer

Rishi's progress has two distinct parts. First, a fixed blend of a THINK+mixer parent and a restricted dynamic-hypergraph network changes the upper-tail stock selection and produces stronger historical portfolios than either alone. Second, hand-coded volatility/exposure rules reduce the risk of those portfolios without changing the trained forecasts. The reported Sharpe above 1 is therefore not evidence that a new neural network suddenly became a better predictor.

The strongest currently supported explanation of the **latest improvement** is risk allocation: less exposure in volatile conditions, somewhat better issuer diversification and lower trading activity. The original model blend may contain useful time-varying information, persistent stock preferences, familiar risk exposures, or a combination. The available evidence does not yet identify their shares. Historical selection luck remains a competing explanation. These explanations can coexist.

Two findings deserve equal attention:

- The dynamic network's reported ten-seed RankIC is **0.03072**, versus **0.00377** for its parent. This is a genuine lead in the committed summary, worth explaining rather than dismissing because other graph studies failed.
- Its blend's RankIC falls to **0.01889**, yet top-five Sharpe rises to **0.89590**, from dynamic-only **0.65662** and parent-only **0.53144**. The mechanism cannot be inferred from average RankIC alone. The upper-tail intersection, identity preferences, turnover, exposure and realized stock returns need inspection.

**Recommended direction:** GO on a bounded mechanism/attribution program using existing predictions. DEFER a claim of learned relational alpha, new architecture expansion, or untouched confirmation until the required artifacts and controls exist.

## 2. What was actually inspected and tested

### 2.1 Evidence levels

| Evidence | Status in this investigation | What it establishes |
|---|---|---|
| User images and Discord text | Read as supplied claims/context | Rishi explicitly says the >1 update changes strategy, with no new training |
| Canonical code, frozen protocols, source-reported audits | Directly inspected | What the implementation computes and what its authors report having verified |
| Twelve committed drawdown-study daily paths, 1,003 sessions each | **Independently recomputed aggregates** | Published SR/CAGR/volatility/maxDD agree within **5.55e-14**; not an order-level replay |
| Historical QQQ/SPY paths | Inspected and aligned to the same dates | Descriptive benchmark/spanning analysis, including entry/terminal fees |
| Parent mixer and pinned dynamic adapter | Synthetic forward probes | Capacity, mask behavior and arithmetic properties at random initialization |
| Three focused test modules | **18 tests passed** | Synthetic timing, normalization, masking, parent initialization, accounting, risk-causality and rank-blend checks |
| Trained newest forecasts/checkpoints, full accounts and raw NDX inputs | **Unavailable in this checkout** | Trained signal, exact stock-level attribution and independent raw replay remain unresolved |

The root `runs/` directory, assembled NDX manifest and architecture-reference cache were absent. I recovered only the two pinned upstream dynamic source files through the GitHub connector, into this study's `source_snapshot/`; I did not fabricate trained predictions or populate the frozen cache. Their content matches the frozen SHA256 records after terminal-newline normalization; both saved and frozen hashes are disclosed in `SMOKE_RESULTS.json`.

Existing source audits report much more extensive checkpoint/account replay. Those remain **source-reported** verification, distinct from the work performed here. The earlier audit's missing-artifact limitation is not erased by successful metric recomputation.

### 2.2 Recomputed ten-seed accounts

All values below were recalculated from the committed daily net paths. SR uses the source's fixed 3% annual hurdle; idle cash earns zero. CAGR is compounded, not annual arithmetic mean.

| Account | Sharpe | CAGR | Realized annual volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Full-exposure top-five blend | 0.896 | 31.87% | 34.05% | -40.94% |
| Original selections + 20% target + MA200 cap | 1.081 | 25.50% | 20.16% | -20.84% |
| Five issuers + 20% target | 1.112 | 26.81% | 20.63% | -21.69% |
| Five issuers + 25% target + MA200 cap | 1.085 | 28.70% | 22.96% | -22.98% |

The original-five **1.199 SR** headline belongs to the original selections with a 20% volatility target, not to the ten-seed issuer account. Its metric is present in `master_update/leader_metrics.csv`; that precise path is not one of the twelve exported `comparison_curves.csv` accounts, so this session does not claim to have independently recomputed it. Original-five and new-five use different rank ensembles; the ten-seed portfolio is not the arithmetic average of their return paths.

### 2.3 Descriptive exposure and risk-control findings

OLS on daily excess returns with intercept, a fixed 3% hurdle and Newey-West lag 5 gives:

| Ten-seed account | QQQ beta | QQQ regression R² | Annual arithmetic intercept | Intercept t |
|---|---:|---:|---:|---:|
| Full-exposure blend | 1.300 | 0.816 | 16.07% | 2.21 |
| Five issuers + 20% target | 0.746 | 0.732 | 14.65% | 2.69 |
| Five issuers + 25% target + trend | 0.812 | 0.701 | 15.90% | 2.58 |

This is **conditional, selected-history, single-factor attribution**, not a causal estimate, investable hedge, factor-adjusted discovery or search-corrected significance result. QQQ returns include small account-entry/exit costs. FF5/MOM/ST reversal, sectors, nonlinear exposures and stock identity have not been controlled here. Do not interpret R² as the percentage of profits explained: it concerns return variation.

Against its seed-group-matched **original top-five full-exposure baseline**, the issuer/20% overlay has beta **0.565**, annual intercept **5.69%**, t **1.53**; the issuer/25%+trend account has beta **0.629**, intercept **5.72%**, t **1.41**. These comparisons change issuer allocation and exposure together. They are compatible with an improved combined policy beyond constant linear rescaling, but do not isolate exposure timing; that requires the allocation-matched `issuer5/full` path, not present in this export. Source paired SR intervals include zero: approximately **[-0.130, +0.542]** and **[-0.146, +0.518]**, respectively, also versus the original full baseline.

Existing committed component controls already show constant 75% exposure at ten seeds has SR **0.863**, versus **0.896** full exposure; 20% basket-vol targeting without issuer selection has SR **1.080**, versus **1.112** with issuer selection. These are source-table comparisons, not separately replayed paths here. They suggest the volatility rule is the larger contributor and that simply holding some cash is not a complete explanation. They do not identify the exact causal share of each component.

### 2.4 Synthetic findings that change the research priorities

1. **The parent can rank identities without different histories.** Twelve identical histories on a symmetric graph produce effectively identical THINK-core scores (cross-stock SD ~4.53e-19), but SD **0.20354** after the N→16→N mixer. Its security-indexed biases and weights are sufficient to create persistent preferences. This is an architecture capability at random initialization, not evidence that trained holdings are a static AI-stock bet.
2. **The dynamic family also has identity capacity.** Two stock-specific diagonal wavelet filters exist. Identical inputs yield random-init score SD **0.000538**; setting both filters to 1 reduces it to ~**1.82e-7** on the symmetric toy. Identity control must cover both model families, not just THINK's mixer. Absolute score magnitudes are not comparable across families.
3. **The dynamic edge aggregator uses sums.** Incidence is a stock-wise softmax over 64 edges; edge representations are `H.T @ X`, not degree-normalized means. Duplicating an embedding population doubles these pre-normalization sums. Final predictions need not double because nonlinearities/LayerNorm follow. Membership count/composition and missing-data patterns could matter; test them rather than assuming a changing hyperedge is economic group discovery.
4. **The imported dynamic forward path matches upstream on all-active synthetic inputs**, and changing inactive inputs to NaN does not change eligible predictions in the tested case. This is reassuring but not a full mask/cold-start audit.
5. **The AH+PM head starts at zero**, so altering auxiliary values at initialization cannot change the score. It trains jointly afterward. Inference branch removal later isolates the direct correction only; it cannot undo changes learned by the jointly trained core.
6. **“MSE” is an incomplete description of both networks together.** Parent training uses masked MSE; dynamic training uses per-date `sqrt(MSE + 1e-8)`. Its gradient is the date's MSE gradient divided by `2*sqrt(MSE+epsilon)`. The toy ratio is 0.22361 for MSE=5. This changes relative date weighting and optimizer trajectories; there is no proof it explains the performance gap.
7. **The sparsified wavelet matrices are not exact inverses.** The toy maximum `|wave @ inverse - I|` is **0.27446**. The code zeros values below +1e-5, including negative inverse entries. Consequently, uniform diagonal filters leave a graph operator; they do not rigorously remove propagation. This is an inherited transform choice to document and isolate, not grounds to declare the historical account invalid or silently change a locked experiment.

## 3. How the current system produces an account

### 3.1 Data, labels and timing

- Historical monthly NASDAQ-100 membership, retaining former members/share classes; 2022-2025 development evaluation on **1,003 shared sessions**. This is distinct from Tom's PIT S&P 500 tests and Ethan's minute-frequency system.
- Five daily channels: return, distance from 5/20-session price averages, 20-session volatility, relative dollar volume. Twenty AH/PM channels: ten per session, including returns/activity/coverage/age summaries.
- Sixteen-session windows. Numeric extended-hours features need at least four active 15-minute bars. Sparse values use train-only mean imputation; coverage/age indicators remain observable. Only about half of member sessions have sufficient active bars. The coverage pattern itself could encode size, liquidity or identity.
- The parent keeps a five-channel THINK core and a separate flattened 320→16→1 AH+PM branch. Dynamic consumes all 25 channels through a common encoder. Therefore their uses of the “same inputs” differ architecturally.
- Information cutoff 09:15 ET; forecast/order freeze 09:20; regular-close fill. PM is the next **exchange** session, not the next calendar day. Core size/NAV/ADV use the prior close. Five-session labels begin at the entry close; earlier AH/PM moves are not credited as earned return. A vendor availability lag is assumed rather than established from first-publication records.
- Train/validation/test are chronological annual folds with outcomes purged across boundaries; prior-year raw-return validation MSE chooses checkpoints. Graphs and scaling derive from training rows. No test-outcome eligibility mask should enter inference.

### 3.2 Two different networks

**Parent:** equation-based THINK reconstruction → train-derived positive-correlation groups → hyperbolic temporal/spatial transformations → scalar forecast plus a stock-indexed mixer. The AH+PM branch adds a centered tanh correction scaled to half the detached core score SD. The parent uses AdamW, LR .001, decoupled weight decay .001, eight-date updates, 30-epoch cap/patience six.

**Dynamic:** pinned SPF author architecture transferred with a restricted snapshot. A 25→32 encoder, temporal attention and intensity/decay processing feed three fused paths: own-stock representation, learned 64-edge aggregation and two static wavelet convolutions. Softmax fusion weights are learned but global; their magnitude alone is not contribution because branch scales and downstream nonlinearities differ. Static input is one training-correlation snapshot, replacing the authors' richer snapshots. Adam LR .0001, per-date RMSE, 30-epoch cap/patience six. This is not a full reproduction of the SPF paper or proof the learned hyperedges caused the gain.

**Critical confounds:** width, feature integration, temporal processing, stock-specific filters, loss, optimizer, update batch and relation representation change together. A performance gap between the current families identifies a package, not the value of dynamic relations.

**Exact implementation matters:** pinned attention computes a square-root dimension scale, then takes its square root again in the divisor (fourth-root scaling). Preserve this source behavior for reproduction. Textbook square-root scaling, an identity static operator or an unsparsified inverse are separately declared future variants, not silent repairs. The smoke's twelve-node model has 25,157 parameters; the report's larger security universe has 25,445. Two node-indexed filters account for the size dependence; the toy is not the trained market model.

### 3.3 Blending and execution

For each origin, transform each seed's eligible-stock predictions to percentile ranks, with average ties. Average ranks across the fixed seeds within a family. Then compute `E = 0.5*parent_family + 0.5*dynamic_family`, without re-ranking either family average first. The two networks are not averaging raw forecast values or holding two separately funded portfolios.

Primary historical comparison used ten seeds and ten holdings. The quoted ~0.919 is the selected original-five/top-five cell; the new risk study applies overlays to top-five selections. Crossings at this hard boundary can have large returns even when whole-universe RankIC changes little. Seed ensembling may reduce noisy rankings or reinforce stable common preferences; both need tests.

Equal-weight targets become frozen share orders, subject to 1% prior ADV trade caps, fill cancellation, cash scaling, corporate settlements and terminal liquidation. The fee formula is `7 bp + 3 bp*sqrt(participation/1%)` per traded side at base cost, i.e. 10 bp at 1% participation, not Ethan's quoted .001/.25 bp experiments. Portfolio labels and target weights are not actual post-fill positions.

Risk overlays compute the proposed basket's trailing 20-session volatility including historical cross-stock covariance, cap exposure at 100%, optionally cap it at 50% below QQQ's MA200, and hold remaining cash at zero interest. There is no leverage. The latest search tested **305 recipes, 915 base-cost accounts and 54 extra cost-stress accounts**. Six recipes met its historical screen. A before-outcome grid freeze on previously inspected years does not make those years untouched.

## 4. Competing hypotheses and decisive observations

| Hypothesis | Why plausible | Observation that would support it | Observation that would weaken it |
|---|---|---|---|
| Persistent identity/style exposure | Security-indexed mixer/filters; concentrated portfolio; beta 1.30 | Training-only mean predictions/labels recreate ranks and net account within a narrow equivalence margin | Large reproducible time-varying increment beyond those controls |
| Own-stock reversal/momentum/volatility | Inputs directly contain these patterns | A small train-only surrogate explains rankings, switches and P&L | Residual neural scores add robust tail information after causal controls |
| AH/PM information continuation or liquidity state | Channels expose activity/returns before entry | Matched trained price/activity feature ablations and delayed-input tests isolate an increment after entry | Coverage-only/lagged-history controls perform similarly; full-sample AH increment remains uncertain |
| Useful contemporaneous peer information | Dynamic branch aggregates related stocks | Relational synthetic recovery and matched learned-graph versus no/frozen/placebo graph increments | Shared encoder/no-peer or matched identity controls explain performance |
| Ensemble upper-tail intersection | Blend outperforms stronger standalone RankIC model | Agreed-high or promoted names have incremental future returns under identical execution | Gains dominated by a few persistent winners or unstable boundary flips |
| Risk forecasting / exposure timing | No retraining for >1 SR; lower vol/beta | Same holdings with causal exposure outperform constant/risk-matched controls; rule helps simple books too | Only neural selected recipe works; gains vanish under timing/cost checks |
| Selection and sample luck | Many strategies, k/seed choices, 305 overlay recipes | Broad uncertainty, weak new-five, concentration, unstable interventions | Frozen future candidate and mechanism contrasts reproduce after costs |
| Data/accounting artifacts | Vendor vintage, membership, cold-start and actions remain imperfect | A timing/identity/settlement repair materially changes relevant accounts | Independent point-in-time replay and publication-latency checks match |

These are not mutually exclusive. Seek a decomposition with uncertainty, not a single attractive story.

## 5. Phase A — make the evidence replayable

**Priority:** mandatory; no new fitting. **Outputs:** `ARTIFACT_INVENTORY.json`, `REPLAY_REPORT.md`, `REPLAY_ERRORS.csv`, `PREREG.md` for the new diagnostic batch. Proposed implementation: `research/analysis/rishi_mechanism_artifacts.py`, tests `tests/test_rishi_mechanism_contract.py`.

- [ ] Obtain existing exports, without requesting new model training or sending messages automatically. Required roots: `runs/ndx_reference_networks_20261024/`, `runs/ndx_extended_hours_20261024/after_hours_premarket/`, `runs/ndx_drawdown_20261026/batch1/`, reviewed assembled daily inputs, extended-hours feature cube and both pinned author source files. Preserve original hashes/paths in the inventory; portability rewrites belong in a separate adapter.
- [ ] Each prediction row must identify family, annual fold, seed, selected epoch, origin/cutoff/decision/fill timestamps, stable security ID, eligible mask, label-observation mask, unrounded forecast, training/validation/test index and input/scaler/checkpoint hashes. Include training and validation predictions for identity controls, not only test predictions.
- [ ] If training predictions were not exported, regenerate them by inference from the original frozen checkpoints and preprocessing, with no optimizer updates or model selection. Label them in-sample descriptions; absence of this export alone does not justify retraining. Missing checkpoints or required inputs still block regeneration.
- [ ] Each account must provide target scores/weights, frozen share orders, fills/cancellations, shares, marks, cash, fees, corporate settlements, ADV, daily NAV, actual exposure and issuer mapping. Include the full 305-recipe inventory and failures, parent/dynamic standalone/blend paths at k5/k10, both halves and ten seeds.
- [ ] Independently derive twelve known paths and the 1.199 original-five leader from orders/holdings. Require exact IDs/calendar and ranking/tie behavior; forecasts agree to predeclared numeric tolerance (float32 absolute 1e-6 pp), daily returns within 1e-10 and normalized NAV within 1e-8 relative. If hardware differences breach forecast tolerance, quantify rank/order changes before amending it; do not widen a tolerance simply to pass.
- [ ] Test duplicate IDs/dates, wrong axes, missing eligible forecasts, changed hashes, inactive NaNs, insufficient label history, split/rename/settlement dates, terminal fees, future-bar mutation, and DST/early-close/weekend alignment. Recompute rather than trust a stored “passed” field.
- [ ] Verify membership snapshot formation was available before the decision, every normalization/group uses the right training boundary, PM ends by cutoff, and actual target labels are absent from inference. Obtain first-publication/vintage evidence if possible; state explicitly if only retrospectively assembled bars exist.

**Gate:** any unreconciled valuation, chronology or identity defect → STOP mechanistic interpretation of affected accounts pending repair. Missing payload → DEFER that analysis, retaining source-summary status. Aggregate matches alone do not pass the raw-replay gate.

## 6. Phase B — separate stable preferences from changing information

**Outputs:** `SCORE_DECOMPOSITION.csv`, `STOCK_ATTRIBUTION.csv`, `IDENTITY_REPORT.md`, saved control accounts. Proposed code: `research/analysis/rishi_mechanism_scores.py`; tests `tests/test_rishi_mechanism_scores.py`.

### 6.1 Define the controls before scoring

Use the same fold, eligibility, stock IDs, decision/fill clock, ranks, costs and original seeds for every arm. Ten seeds/top ten retain continuity as the primary mechanism comparison; top five and the two existing risk leaders remain explicitly selected diagnostics.

- **B/D/E:** original parent, dynamic and fixed blend.
- **H-label:** each ID's mean five-session label from that fold's training rows with `labelmask=True` and finite labels, at least 30 such observations. Ignore zero-filled invalid labels. Replace sparse/unseen IDs with the training cross-sectional median among IDs meeting that count; if none qualify, mark the control unavailable rather than use test information.
- **H-pred:** each ID's mean training prediction from `eligible=True` finite rows, at least 30 observations per seed/family, then combine with the same rank rules. Ignore inactive zero-filled forecasts; use the corresponding qualifying-training-ID median for sparse/unseen IDs and record support counts. Regenerated predictions are in-sample descriptions. This diagnoses learned stable score preference; H-label diagnoses a cheap historical-return substitute. They are different controls.
- **E-H:** retain the parent and replace dynamic with ranked H-label; also report a labeled H-pred substitution. No fitted blend weights.
- **Changing component:** subtract each ID's frozen training-prediction mean from test scores before ranking; center using only training statistics. This is a diagnostic that may change calibration, not an already trained counterfactual model.
- **Simple controls:** train-only ridge using the five daily channels, REV5, momentum126, coverage/activity-only, and EW eligible universe. These are fixed diagnostic comparators, not a new tuned model tournament.

**Availability rule:** retain the original B/D/E tradable mask. Compute comparator signals solely from completed history available at that origin. If a signal is unavailable, assign a neutral score equal to the current eligible cross-sectional median of finite comparator scores. If no comparator scores are finite, use a tied neutral score for all eligible names and flag that origin; source tie-allocation rules apply. Freeze this fallback before evaluation, retain support/coverage flags, and report a separate common-signal-availability sensitivity selected only by ex-ante inputs. Never filter tradable names using future label availability. Restrict every model's *ranking metric* to a common observed-label mask, kept distinct from the trading mask. A control dominated by fallback rows is diagnostically weak, not evidence its underlying signal fails.

### 6.2 Measure behavior and economic contribution

- [ ] Measure within-ID and between-ID score variation with fold/membership accounting; rank autocorrelation; top-k membership persistence; turnover; concentration by stable issuer; ID entry/exit effects. Do not pool fold-specific scales indiscriminately.
- [ ] Tabulate additive marked-dollar P&L divided by preceding NAV for every security and day, with fee and settlement reconciliation. Separately report log-growth attribution if a linking method is used; never sum per-stock CAGRs as a portfolio contribution.
- [ ] Report the top 1/3/5 issuer contributions and all others, without pre-assuming NVDA or another name dominates. Fixed observed-name deletion is post-hoc influence analysis; a retradable exclusion must be determined from training data only.
- [ ] Relate training-only stock preferences to ex-ante beta, momentum, volatility, liquidity and dated sector/issuer attributes. Distinguish time-series QQQ loading from cross-sectional stock selection. Use current-vintage factors only for labeled ex-post attribution.
- [ ] Evaluate paired per-origin RankIC, top-k minus eligible EW label return, decile spreads, net share-account returns and active-return IR. Cluster inference in market time; stocks/seeds are not independent samples.

**Interpretation rule:** close performance is not proof of equivalence. Use the earlier proposed economic margin ±2 pp/year of paired net arithmetic mean return, with a synchronized 95% interval. Contained interval → economically similar conditional on this history; wide interval → DEFER; clear residual advantage → proceed to time-varying mechanisms. Also report paired SR and risk differences. No “identity explains everything” claim from a high correlation alone.

## 7. Phase C — identify why blending changes the upper tail

**Outputs:** `BLEND_BOUNDARY_REPORT.md`, per-origin overlap/promotions table, `BLEND_CONTROLS.csv`. Proposed code: `research/analysis/rishi_mechanism_blend.py`; tests `tests/test_rishi_mechanism_blend.py`.

- [ ] Reproduce percentile-rank → within-family seed mean → 50:50 family mean exactly, including average ties and inactive names. Test that raw score scale changes do not affect the rank ensemble and that no extra family re-ranking slips in.
- [ ] Treat the reported raw forecast-error correlation near .998 cautiously: when forecasts are small, both errors share the much larger negative realized return. Inspect forecast covariance/correlation and residual errors relative to common baselines alongside tail-selection differences; the .998 statistic neither proves nor rules out useful complementarity.
- [ ] Compare parent-only, dynamic-only and blend at the same k and policy. Identify agreed-high names, parent-promoted names, dynamic-promoted names and displaced names. Report ex-ante scores, subsequent entry-to-exit returns, costs, exposure and issuer concentration; do not select a profitable category afterward.
- [ ] Compare the rank-blended account with a **50:50 capital split between the two standalone accounts**, with capital-aware sizing/costs. The difference separates changed stock selection from ordinary return diversification. A simple average of full-size NAVs is only an algebraic diagnostic if costs/caps depend on AUM.
- [ ] Repeat the fixed H-label/H-pred replacement. Ask whether dynamic contributes new changing information or merely steers a different persistent top-tail basket.
- [ ] Analyze the original/new halves and predeclared seed bundles (use a fixed RNG and fixed bundle list before evaluation). Report conditional initialization sensitivity; do not search for a stronger seed ensemble or count bundles as independent years.
- [ ] Train a compact explanatory surrogate on training predictions only, with ID intercepts and fixed reversal/momentum/volatility/activity terms. Evaluate held-out forecast/rank fidelity, boundary choices and account fidelity. High forecast fidelity supports a compressed behavioral description, not economic causation; low fidelity does not establish sophisticated information.

**Gate:** if blend advantage is economically replicated by persistent controls, pursue a simpler account/exposure explanation. If it retains changing upper-tail information beyond controls, proceed. Instability with wide intervals → DEFER rather than automatic new training.

## 8. Phase D — separate price information, activity metadata and peer information

**Outputs:** `PERTURBATION_PROTOCOL.md`, `PERTURBATION_RESULTS.csv`, `MECHANISM_MAP.md`. Proposed code: `research/analysis/rishi_mechanism_interventions.py`; tests `tests/test_rishi_mechanism_interventions.py`.

### 8.1 Saved-checkpoint interventions: sensitivity, not causal proof

Freeze the intervention list and donor selection before using returns. Each uses original eligible sets, common seeds and train-derived distributions.

1. Replace daily channels, AH numeric channels, PM numeric channels, and activity/coverage/age channels one block at a time with training-derived typical trajectories; keep imputation/coverage indicators consistent. A zero is a standardized training mean for numeric channels, not an ordinary “no information” state for every metadata feature.
2. Use earlier same-ID windows matched on available volatility/activity to perturb temporal novelty without replacing all identities. Donors must precede the origin and not overlap the recipient label. Report donor distances/support; unfamiliar combinations can create out-of-distribution failures.
3. Shift the AH/PM stream by one earlier exchange session as a fixed timeliness diagnostic; recompute the alignment, not just array positions. Compare direct parent auxiliary removal with a separately retrained daily-only control later, because joint training changes the core.
4. For peer sensitivity, preserve a focal stock's own full history and intervene on other stocks using eligible, historically matched donor windows. Preserve broad market state/activity as far as feasible and measure residual distribution shift. “Shuffle stock features but keep labels” changes identities and relations together; it cannot by itself prove peer causality.
5. Log dynamic incidence, entropy, edge masses, effective edge count, temporal stability, and co-assignment `H H.T`. Align edge labels across days or use permutation-invariant co-assignment; raw hyperedge numbers are exchangeable. Check whether assignments mainly track beta/liquidity/coverage or stable ID preferences. Attention and gradient maps are descriptions, not contribution estimates.
6. Inspect frozen learned diagonal filters, mixer weights and training-derived snapshot spectra. Compare them across seeds/folds without using their aesthetic appearance as evidence. Test real versus randomized peer identities and no-propagation interventions separately from an ID-filter intervention.
7. Re-run member-count/masking toys at several eligible counts with inactive values NaN, finite garbage and zeros. Verify membership gates prevent irrelevant nodes from influencing eligible scores. Distinguish true population changes from artificial replication; summation scaling is a sensitivity to test, not inherently an error.
8. Decompose saved-checkpoint representations explicitly: parent core/mixer/AH correction and dynamic temporal/learned-edge/wavelet branches, including their scales and downstream effects. Measure trained parameters' changes from initialization by supported ID. A consistent ID-reindexing audit must jointly permute feature rows, masks, static operators and stock-indexed filters/mixer input-output parameters. This should preserve economic predictions after undoing the permutation. Feature-only swaps answer a different, distribution-shift question. Track trained, sparse-history, newly eligible and unsupported IDs separately, including inactive gradients and eligibility transitions.

Primary checkpoint outputs are forecast/rank changes, top-k switches and the paired account changes under identical execution. A perturbation may establish that a feature/path affects a model. It does not establish that the feature carried unique predictive information or that the original model learned an economic cause.

### 8.2 Only if necessary: matched retraining that identifies mechanisms

Do not run a broad architecture grid. First run synthetic learning gates for own-stock signal, persistent identity, useful time-varying peer signal, identity-only/no-peer null, missingness-only nuisance and changing membership. Include a train/test process change to detect memorization. Each control must recover the planted signal it is designed to represent with comparable optimizer budget; failure makes a financial null uninterpretable.

**Stage D1: dynamic relation × identity 2×2**, only after saved-output evidence leaves both plausible. Hold the 25-channel encoder, temporal blocks, optimizer, RMSE, widths, validation rule, seeds and budget fixed:

- Learned-dynamic relation on/off: dynamic peer messages versus a capacity-matched own-stock path, keeping the static snapshot branch fixed. This isolates the learned-dynamic path; it is not an all-relations/no-relations comparison. A separate no-propagation static operator is required before claiming all peer information helps.
- Explicit stock-indexed parameters on/off: stock-indexed wavelet filters versus shared filters, preserving the rest of the head; explicitly handle N-indexed parent mixers in a separate parent control. The fixed graph can still encode identity/structural position, so shared filters are not an identity-free architecture.

This identifies effects of these particular mechanisms, including their interaction; it does not establish that every possible dynamic graph helps. Fit a frozen-incidence and degree/cardinality-matched random-relation control only if relation-on has a credible increment. Keep static-wavelet and learned-incidence questions separate. Equal seed/validation/compute budgets are mandatory; parameter matching alone does not ensure equal learner power.

**Stage D2: minimal feature experiment**, chosen from Phase D sensitivity results before any new financial fits: daily + activity-only versus daily + full AH/PM, with a shared architecture/parameter inventory and a daily-only baseline. If PM specificity remains unresolved, add exactly one declared PM timing contrast; do not cross every channel with every graph/loss.

**Stage D3: loss/optimizer control**, only if the graph mechanism is still confounded: compare MSE/RMSE under the same dynamic encoder and optimizer, with equal training budget and validation criteria. This tests date reweighting, not whether “MSE is better than ranking” in general.

Each stage needs its own committed preregistration, synthetic gate, fixed financial batch and STOP/DEFER/CONTINUE rule before scoring. All 2022-2025 results remain exploratory. Log CPU/GPU seconds per epoch and memory on synthetic data first; project fit counts × measured epochs × time and seek a resource decision before launching a long job. No unmeasured promise of training time.

## 9. Phase E — explain the >1 Sharpe independently of the neural story

**Outputs:** `RISK_ATTRIBUTION_REPORT.md`, exposure/selection factorial accounts, cost and calendar attribution. Proposed code: `research/analysis/rishi_mechanism_risk.py`; tests `tests/test_rishi_mechanism_risk.py`.

- [ ] On the same fixed blend rankings, replay selection `{original top5, issuer5}` × exposure `{full, constant .75, basketvol20 target .20, MA200 cap .50, target .25 + MA200 cap}`. These are fixed existing controls, not a search for a new winner. Report their interactions; do not assign additive “percent of Sharpe explained” because SR is nonlinear.
- [ ] Apply the **same fixed 20% volatility rule** to parent-only, dynamic-only, REV5, H-label, EW and QQQ. Define the basket-return history, warmup, weekly clock, no leverage, fill timing, participation costs and cash policy in advance for every arm. This tests whether the improvement is a broadly useful risk overlay rather than information unique to the neural ranker.
- [ ] Include a training-calibrated fixed-exposure comparator, frozen before each fold's outcomes, targeting similar predicted risk. Full-sample equal-realized-vol scaling is an ex-post algebra check, not a tradable control. Constant zero-cost scaling with cash rate=rf and rf=0 preserves Sharpe; that identity does not carry over automatically to zero-interest cash, a 3% hurdle, drifting shares and trading fees.
- [ ] Decompose average exposure, actual beta, trading fees, volatility, downside covariance and arithmetic return. With a fully reconciled frictionless base sleeve, use `mean(e*r) = mean(e)*mean(r) + cov(e,r)` descriptively; reconcile the actual account residual from fills/drift/costs and equity exposure timing. A good covariance on selected history is not proof of return forecasting.
- [ ] Attribute cumulative log-growth and risk changes to calendar years, with exact calendar return `prod(1+r)-1`. Annualized subperiod returns differ when the number of sessions is not 252. Display the 2022 drawdown episode and all later years; no retrospectively selected market regimes.
- [ ] Compare 1×/2×/4× participation fees and separate 10/20/30 bp-per-side scenarios. Retain cap/cash feasibility; do not replace a fee formula in a frozen account and call it identical. Add Treasury cash yield only as a clearly separate accounting view. Report borrow/margin costs if any future hedge diagnostic is used.
- [ ] Audit how quickly a sudden downside event can affect trailing volatility and the next close fill. Test gaps, missing marks, new entrants, low history, canceled orders, changed issuer mapping and drift. A 20% target is not a 20% drawdown stop or hard risk bound.

**Gate:** a broadly beneficial overlay can be useful even if relational alpha fails. A neural attribution claim requires residual selection information beyond risk/identity controls. Negative overlays do not by themselves refute the model's forecast information. Preserve these separate verdicts.

## 10. Statistical design and decision rules

All new historical diagnostics need a frozen list of primary contrasts and complete failure reporting. The source's 305-recipe search and earlier 346-strategy comparison are related, dependent searches; do not count them as independent draws or infer an expected maximum from that assumption. Within-batch multiplicity adjustment cannot repair the full prior search.

**Proposed new mechanism family:** E−B and E−E-H at ten seeds/top ten under the unchanged full-exposure policy. Scalar endpoint for each is `252 * mean(daily_net_return_left - daily_net_return_right)`. Null H0: annual net arithmetic mean difference ≤0; one-sided Holm family level 5%, using the synchronized date-block null distribution with contrasts centered at zero. Report paired two-sided 95% intervals as well. This new diagnostic endpoint emphasizes economic attribution; it does not rewrite the earlier team protocol's primary SR test. SR differences remain secondary here and must be evaluated separately under any retained earlier prospective protocol.

**Secondary attribution:** selected top-five risk leaders versus both original and allocation-matched full exposure, plus causal fixed-risk controls. Freeze those families separately; the newer higher-SR row does not retroactively replace the earlier primary candidate.

- Use paired daily account differences and synchronized date resampling across all arms. Proposed new-batch primary: 10,000 stationary-bootstrap draws, expected block 21 sessions; fixed 5/63 sensitivities. Preserve source circular-block-20 results as source diagnostics; do not mix them without labeling the method.
- Report 95% paired intervals for net arithmetic mean difference, SR difference, CAGR difference, active-return IR and exposures. Use Holm 5% for the two primary one-sided mechanism tests; emphasize interval widths. Per-origin ranking inference uses non-overlapping five-session origins and respects folds. Do not pseudo-replicate stocks, seeds, channels or duplicate labels.
- Report NW mean/factor statistics with declared lag 5 and lag 20 sensitivity. They are supportive attribution, not a substitute for paired account tests. Single-factor selected-history t statistics from this smoke test are not promotion gates.
- A claim of near-equivalence requires a 95% interval contained in ±2 pp/year net arithmetic mean difference. Insufficient precision is DEFER, not proof of no difference. This margin is a proposed resource threshold, not a discovered financial law.
- **STOP interpretation:** unresolved timing/accounting/identity defect. **DEFER:** missing raw evidence, wide mechanism intervals, failed learner-power gates or unsupported execution. **Proposed GO to financial retraining:** raw replay passes; both primary contrasts reject their H0 under the declared family procedure; both point estimates are at least +2 pp/year net arithmetic mean at base costs and positive at the separate 20 bp/side view; and all required synthetic learning gates pass. Otherwise continue descriptive saved-output work only, or DEFER. If either contrast has an upper 95% bound below +2 pp/year, STOP additional complexity spending for that stated economic threshold, without claiming the model has no information; an upper bound ≤0 is stronger evidence against that particular increment. A narrow equivalence result favors simplifying the mechanism. These are proposed research-resource rules to freeze before the full diagnostic batch, not confirmation of future alpha or retroactive tests of this smoke work.

Estimate power only after the paired tapes exist. Simulate 1/2 pp annual net-mean increments and .10/.20 SR increments using paired long-run dependence, with calendar/seed uncertainty kept distinct. Before any prospective test, choose a fixed endpoint with stated power; if a two-year horizon cannot identify the desired effect, declare an exploratory observation period or a longer fixed endpoint. Do not extend based on emerging p-values.

## 11. Prospective confirmation and external validity

The current confirmation gate is unactivated. This investigation does not activate it. No retrospective 2026 entries, automation, trading or use of the protected S&P window is proposed as a completed action.

After mechanism gates, freeze **one complete candidate**: data vintage/arrival policy, supported stock-ID universe, membership/issuer mapping, features/scalers, seeds/checkpoints or scheduled refit policy, ranking/blend, k, overlay, warmup, costs, cash and failed-input behavior. Choose the candidate by the scientific question and operational robustness; its original-five 1.199 SR does not automatically justify selection over the ten-seed primary.

Collect timestamped forecasts before decisions, append-only orders/positions and common comparator paths. If input availability fails, use one predeclared common fallback across arms; never replace a missing model with whichever control later performed better. Specify unseen-stock/cold-start behavior: random appends to security-indexed weights cannot count as a validated continuation.

Previously proposed November 2, 2026 is merely an earliest possible date after readiness; if a complete freeze is later, start later without backfill. Earlier strategy-specific reservations do not make a newly selected overlay prospective from those earlier dates. Preserve the existing S&P 2024-01..2026-09 restriction separately from spent NDX dates.

Report after a predeclared endpoint, with data-quality monitoring separated from performance-based optional stopping. One year can be badly underpowered. A 252-session diagnostic followed by a 504-session final read must declare the first diagnostic-only or use a valid sequential design chosen beforehand. External replication on another universe is a transfer test, not an untouched NDX continuation; do not overwrite the earlier correlation-HG S&P STOP or treat it as proof against this different architecture.

## 12. Execution order, deliverables and budget discipline

| Order | Work package | Required input | Stop point / output | New financial fits |
|---|---|---|---|---:|
| 0 — done | Source trace, numerical path check, synthetic probes, focused tests | Committed sources/curves | This plan, smoke JSON and diagnostic CSVs | 0 |
| 1 | Artifact contract and independent raw replay | Original forecasts, inputs, trades, checkpoints | Replay gate; inventory/errors report | 0 |
| 2 | Identity, simple-signal and P&L decomposition | Training/test predictions plus positions | Mechanism shortlist; bounded uncertainty | 0 neural fits; only declared simple controls |
| 3 | Blending and risk counterfactuals | Reconciled forecasts/marks/accounts | Selection vs diversification vs exposure report | 0 neural fits |
| 4 | Checkpoint perturbation and synthetic learning gates | Original checkpoints plus bounded synthetic tasks | Sensitivity map and learner-power report | Synthetic only |
| 5 — conditional | Small matched retraining stages | Surviving hypothesis and separately frozen batch | Relation/ID/feature/loss result | Budget measured and declared first |
| 6 — separate decision | Candidate freeze and prospective collection | Ready system and powered endpoint | Timestamped future record | Only the chosen refit policy |

The raw export/replay is the immediate bottleneck. Steps 2–4 can be prepared while exports are obtained, but cannot manufacture their missing results. No further team message, paid-data acquisition or long training launch is included in this task.

Final research deliverables should answer, with uncertainty:

1. What fraction of additive realized P&L is tied to which names, dates, costs and exposures?
2. Does changing information improve over a frozen training-only identity score and simple signals?
3. Does blending add selection information or mostly rearrange a persistent upper tail?
4. Are AH/PM price channels useful beyond activity/coverage metadata at the actual entry time?
5. Does peer information help after controlling for encoder, loss, identity and learner power?
6. How much of improved SR/drawdown is reproducible with the same risk rule on simple books?
7. Which mechanism would justify one frozen future experiment, and what would falsify it?

## 13. Source map, reproducibility and review

### Local sources

- Entry points: `charts.md`, `latest/README.md`, `docs/PROJECT_LAYOUT.md`.
- Parent: `research/models/research_models.py` (N→16→N mixer), `research/models/think_model.py`, `research/models/ndx_extended_hours.py` (timing, normalization, bounded auxiliary correction).
- Dynamic: `latest/models/ndx_reference_models.py` (snapshot and masking adapter), `latest/training/ndx_reference_training.py` (RMSE and validation selection), pinned upstream snapshots below.
- Ensemble/evaluation: `research/training/run_robust_research.py::ensemble_predictions`, `latest/reporting/report_ndx_reference_networks.py::blend_scores` and `forecasts`.
- Data/economics: `research/core/pit_research_engine.py` (folds, membership mask, correlation groups, costs and accounts), `research/core/research_engine.py::performance`.
- Risk: `research/core/ndx_drawdown_engine.py`, `docs/ndx_drawdown_20261026/{REPORT.md,PROTOCOL.json,metrics.csv,uncertainty.csv,comparison_curves.csv}`.
- Reports/metadata: `latest/results/reference_networks/{PROTOCOL.json,ranking.csv,prediction_quality.csv,complementarity.csv,metrics.csv,paired_intervals.csv}`, `docs/ndx_extended_hours_20261024/REPORT.md`.
- Earlier plan retained: `docs/team_audit_20261026/HYPERSHIFT_NEXT_EXPERIMENT.md`; this plan extends its mechanism controls to the new risk study rather than treating its missing raw replay as complete.

### External primary sources

- [Pinned SPF author layer code](https://github.com/kijeong22/ijcai2025-spf/blob/9477001ad9b85beb3455234675535e26c3d33e44/src/framework/layer.py) and [model code](https://github.com/kijeong22/ijcai2025-spf/blob/9477001ad9b85beb3455234675535e26c3d33e44/src/framework/model.py): full source retrieved through the GitHub connector and inspected. This verifies implementation details, not the original paper's empirical claims.
- [Moreira & Muir, Volatility-Managed Portfolios, 2017](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf), **FULL TEXT**, especially §I.B–C: motivates a managed-versus-unmanaged spanning comparison. Their main rule scales monthly excess returns by inverse lagged **variance**; Rishi's capped inverse **volatility**, daily estimate, weekly trading and cash/account rules are different. The paper motivates controls; it does not validate these selected NDX results.

No new broad literature claim is needed to answer the code-level question. Independent literature/source review is recorded in `REVIEW.md`, with any tool/model substitutions disclosed.

### Re-run the completed smoke work

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.\.venv\Scripts\python.exe -m hypershift run rishi_mechanism_smoke_20261027
.\.venv\Scripts\python.exe -m hypershift test test_ndx_extended_hours test_ndx_drawdown_engine test_ndx_reference_evaluation -v
```

The first command reads only committed spent-period paths and synthetic tensors. It writes this study's diagnostics, not frozen prior experiment folders. Source snapshots live outside the canonical source/cache and are injected only into the smoke process. The second command passed **18 tests**. The launcher temporarily creates compatibility copies and removes unchanged ones after completion; stage changes afterward.

Files: `SMOKE_RESULTS.json` (input hashes, availability, probes), `RECOMPUTED_PATH_METRICS.csv`, `DESCRIPTIVE_SPANNING.csv`, `ANNUAL_PATH_DIAGNOSTICS.csv`, `PROGRESS.md`, `REVIEW.md`. Preserve raw source/protocol hashes and keep observations, explanations and future hypotheses separately labeled.
