# Audit request (abridged copy of the user's prompt, 2026-10-22)

The full prompt is in the session transcript; the abridgement keeps every instruction that binds the audit.

**Title:** HyperShift — Deep Scientific Review, Methodological Audit, and High-Value Research Roadmap.

**Role:** an independent senior quantitative researcher, asset-pricing specialist, ML scientist, statistician and research-software auditor. Be skeptical. The goal is what is true, not to make HyperShift look successful.

**Rules:**
- Absence of significance is not proof of no effect.
- An exploratory result is not confirmatory.
- A high Sharpe is not predictive skill.

**Parts:**
1. **Review the full history.** Classify each finding as Established / Suggestive / Unsupported / Untested / Compromised. For each major finding give the hypothesis, the experiment, the controls, the effect, the uncertainty and power, the strongest defensible conclusion, alternative explanations and what remains unknown.
2. **What was done well**, and which practices become permanent requirements. Keep reproducibility distinct from validity.
3. **Residual-target finding** (lgbm residual − raw +0.0140 [+0.0018, +0.0261], 5/5 folds, Holm 0.065):
   - A. Is it genuine? Possible explanations: real skill, objective/metric alignment, an artefact of residualising, variance reduction, removing predictable factors, leakage, selection.
   - B. Audit the target construction.
   - C. Is the comparison matched? If not, what is the minimal experiment?
   - D. Is the gain economically meaningful?
4. **Statistical audit:**
   - power per contrast from the actual paired series (residual vs raw, vs reversal16, vs 0);
   - dependence and block sensitivity;
   - CI vs Holm;
   - families;
   - about 2,000 runs of adaptive history;
   - year consistency;
   - equivalence testing;
   - selection bias of the frozen winner of 17 (backtest-overfitting diagnostics).
5. **Data integrity:**
   - dividend-adjustment look-ahead: mechanism, magnitude, rank effect, effect on conclusions, the prospective pipeline, a minimal sensitivity test;
   - survivorship;
   - corporate actions;
   - zero volume and jumps;
   - feature availability and execution timing.
6. **Prediction-to-trading mismatch:** simulator accounting, neutrality, costs, horizon vs rebalance, exposure-matched baselines.
7. **Literature,** verified and with no invented citations: what each paper tested, whether HyperShift tested the same thing, and the differences.
8. **Re-evaluate the negative claims:** features, model class, volume, graphs, horizons, market-neutral, learned hyperedges, more training.
9. **Missing information:** why it matters, whether evidence exists, the cost, whether it could change a conclusion.
10. **Ranked improvements** P0–P3, each with question, motivation, minimal test, data, baseline, metric, design, cost, outcomes, exploratory vs confirmatory.
11. **One focused next study:** hypothesis, design, controls, power, effect threshold, success criterion, stop/go, implementation.
12. **Protect the prospective protocol.** Classify each defect: (1) repairable without a spec change, (2) needs a separately versioned protocol, or (3) invalidates the claim. Never modify v1.
13. **Execution stages:** A review, B recompute, C cheap diagnostics, D recommendations. No large campaigns. Never claim checks that were not run.

**Deliverables:**
1. Critical review.
2. Issue register.
3. Independent statistical assessment.
4. Literature comparison.
5. Roadmap.
6. Experiment specification.
7. Executive memo answering the 10 questions.
