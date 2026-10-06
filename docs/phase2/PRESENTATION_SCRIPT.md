# Script: "Can we reproduce THINK?"

This script goes with the deck at https://claude.ai/artifact/Dxo8iNU7MqBwNT3pLyEE9x (25 slides, about 20-25 minutes). Each slide has a heading, what to say, and the point to land. The same text is also in each slide's speaker notes.

---

## 1. Cover: Can we reproduce THINK?

Today I'll walk through the whole reproduction study:
- what THINK claims
- every experiment we ran to test it
- why each experiment followed from the last
- what we should do next

The short version: we could not recreate the paper's advantage under a fair evaluation. I'll show why I'm confident that isn't just a bug on our side.

**Land:** this is a story with a conclusion and a decision at the end.

## 2. The question

THINK ranks NYSE stocks each day, buys the top few, and reports a higher Sharpe ratio and NDCG than two kinds of competitor:
- its own Euclidean version, called TCONV+DHHAN
- earlier graph models: STHGCN and RSR

The margins are small: 1.18 vs 1.14 vs 1.10. If the result were real, it would justify building on hyperbolic hypergraph models.

So the question was simple: does that advantage survive when we rebuild the model and test it fairly?

**Land:** the claimed edge is small, so it has to be measured carefully.

## 3. The story in five steps

Each step answered a question the previous one raised:
1. We rebuilt THINK. It matches the paper only if the test year picks the model.
2. We checked whether our code was the problem. We found a real bug and fixed it, and still saw no advantage.
3. One variant scored a Sharpe near 2. Random stock picks score about the same.
4. We tested later years with frozen and retrained models.
5. We tested whether it's the benchmark itself, using the original authors' code.

One rule throughout: the model is chosen on the validation year, never on the test year, and every test is set up before we look at returns.

**Land:** the experiments weren't random; each one was forced by the last.

## 4. Step 1: I rebuilt it from the paper

I wanted to know whether THINK works, so the first step was rebuilding it. The authors' code repository is empty, so everything was written from the equations.

What the paper gives:
- the model
- the data: 1,737 NYSE stocks from the RSR dataset
- the train, validation and test years
- the trading strategy

What it leaves out, and every one of these matters a lot:
- how the reported epoch was chosen
- the exact Sharpe formula
- the hyperparameters
- one misprinted attention operator

Wherever we had to guess, we labelled the guess and later tested the alternatives.

## 5. Step 1: scored the paper's way, we beat the paper

If you pick the training epoch with the best 2017 test Sharpe, our THINK reaches 2.40 against the paper's 1.18. Our Euclidean version reaches 1.64 against 1.14.

That's the same ordering, with higher numbers. Our Sharpe is annualised and the paper's formula isn't precise, so compare the ordering more than the exact values.

**Land:** at first glance, a successful reproduction.

## 6. Step 1: the fork, so I tested both options

The paper never says how the epoch was chosen, so there are two options:
- **A:** pick the epoch with the best 2017 test score. That uses the answer key, but the authors' earlier code prints the test score every epoch with no selection rule, so it's plausible.
- **B (fair):** pick the epoch on the 2016 validation year.

Under A we reproduce the paper. Under B:
- THINK drops to about zero, below its Euclidean version.
- Nothing beats simply holding every stock, which earned 1.53 in 2017.

**Land:** the paper's result depends entirely on how the model is chosen.

## 7. Step 1: why the choice matters

This chart is one THINK run over 100 epochs:
- blue: 2017 test Sharpe
- gray: 2016 validation Sharpe
- dashed line: holding every stock

Test Sharpe bounces between 0.6 and 2.9, and validation barely predicts it (rank correlation 0.22). So reporting the best of 100 test scores almost guarantees a big number, from noise alone.

**Land:** "best epoch on test" is selection on luck.

## 8. Step 2: is THINK wrong, or is our code broken?

There were two explanations:
- THINK genuinely has no edge, or
- our rebuild can't learn, so any real signal gets lost.

Returns alone can't separate them. So I built a test where the right answer is known: plant a signal of known strength inside real prices and measure how much the model recovers.

## 9. Step 2: the test found a real bug, and fixing it worked

Before the fix:

| Model | Planted signal recovered |
|---|---|
| THINK | 26% |
| Euclidean version | 32% |

**Cause:** weight decay of 5e-4 inside Adam was 10 to 40 times stronger than the learning signal. It shrank the hyperbolic layers to a nearly constant output.

**Fix:** weight decay 0, plus inputs scaled relative to each window's last price. After the fix:

| Model | Planted signal recovered |
|---|---|
| THINK | 63-84% |
| Euclidean version | 72-77% |

**Land:** our pipeline can find a signal when one exists. That rules out "our code can't learn", and it meant rerunning everything.

## 10. Step 2: rerun with the fix, still no advantage

Full NYSE, 10 seeds per model, epoch always chosen on validation:

| Price scaling | THINK | Euclidean | Hold all stocks | Corrected p |
|---|---|---|---|---|
| Leak-free | 1.69 | 1.44 | 1.53 | 0.43 |
| Paper's (a separate set of 10 runs) | 1.93 | 1.15 | — | 0.074 |

THINK is ahead on average, but not consistently across seeds; our rule needs p below 0.05.

Subtlety: our inputs divide each 16-day window by its own last price, so the two scalings feed the model almost identical numbers. The gap between 1.69 and 1.93 is therefore mostly run-to-run noise (see slide 17).

**Land:** verdict NO EVIDENCE of an advantage, under the decision rule we set in advance.

## 11. Step 3: a Sharpe near 2, four possible causes

One variant, with the ranking loss switched off, scored a Sharpe near 2. That is above the paper, even though its ranking metrics were near random. Before believing it, I listed four explanations, each with its own test:
- **A. A quirk in tie-breaking.** Test: break ties at random.
- **B. Luck plus market exposure.** Test: compare with random portfolios.
- **C. A data or backtest bug.** Test: audit every input and trade.
- **D. Real skill concentrated in the top five.** Test: hit rates, calibration, score gaps.

## 12. Step 3: random portfolios score about as well

We built 10,000 random portfolios: 5 random stocks every day, from the same eligible stocks, under the same rules.
- **Gray band:** the middle 90% of those portfolios, about −0.5 to 2.3.
- **Blue dots:** our five runs, 1.88 to 2.14. They sit at the 88th to 93rd percentile, inside what luck produces.

Why that's plausible:
- 2017 was a strong year: holding everything gave 1.53.
- The model picked high-risk stocks (beta about 1.3).
- Its correlation with next-day returns was 0.005.

No null comparison rejected after correction.

## 13. Step 3: verdict

| Hypothesis | Verdict | Evidence |
|---|---|---|
| A. Tie quirk | Ruled out | Random tie-breaks keep Sharpe at 1.8-2.2 |
| B. Luck plus exposure | Best explanation | — |
| C. Data or backtest bug | Clean audit | — |
| D. Real skill at the top | Unsupported | Near-random hit rates; bigger score gaps don't predict bigger returns |

Caveat: a single year is low-powered. One run's Sharpe could plausibly be anywhere from 0.3 to 3.5, so "no evidence" is not "no signal".

**Land:** that caveat is exactly why we tested more years.

## 14. Step 4: if it's real, it should hold after 2017

Two tests:
1. **Frozen model:** train once on pre-2017 data, lock the weights, trade 2018-2023.
2. **Retrained every year (walk-forward):** for each of 2019-2023, train on the past, pick the epoch on the previous year, trade the next year. This is how you would actually use a trading model.

Data: Alpaca, chosen because it keeps stocks that later delisted.
- 1,647 of the 1,737 stocks matched the original prices, including 418 that later delisted.
- Yahoo was rejected because it only keeps survivors, which inflates results.
- Everything was locked before any 2018+ return was computed.

## 15. Step 4: frozen model

2018-2023, Sharpe:

| | Sharpe |
|---|---|
| THINK, five seeds | 0.10 to 0.86 (mean 0.43) |
| Euclidean version | 0.16 |
| Hold every stock | 0.51 |

Ranking skill: NO EVIDENCE (every corrected p is at least 0.6). The retrained models reproduced 2017 first, so the replication itself was sound.

## 16. Step 4: retrained every year

2019-2023, all years joined:

| | Sharpe |
|---|---|
| THINK | 0.70 |
| Euclidean version | 0.51 |
| Hold every stock | 0.69 |

- THINK's correlation with next-day returns is 0.002, and the corrected p is 1.0.
- After 25 bp trading costs, 4 of 5 seeds lose money.
- Year by year, THINK wins in rebounds (2020) and loses in calmer years (2019). That is the signature of high-beta picks (beta about 1.25), not stock-picking skill.
- Retraining wasn't detectably better than the frozen model.

## 17. Step 4: two identical runs disagree

This is the most important slide. Two runs have identical settings; the only difference is GPU randomness.

| Run | THINK | Euclidean | THINK ahead in |
|---|---|---|---|
| Run 1 | 1.98 | 2.05 | 2 of 5 seeds |
| Run 2 | 2.20 | 1.46 | 4 of 5 seeds |

The paper reports spreads of about ±0.005. Between our seeds the spread is ±0.3 to 1.5.

**Land:** the paper's central comparison (1.18 vs 1.14) is inside the noise.

## 18. Step 5: is it THINK, or the benchmark?

The two explanations predict different things:
- If our THINK is the problem, the original authors' own models should show real skill on this data.
- If the benchmark is the problem, their code should show the same no-skill pattern.

Three tests:
1. the RSR authors' original code
2. STHGCN's official code
3. every reading of the paper's evaluation rules

## 19. Step 5: the original RSR code

RSR is the earlier stock-ranking paper THINK builds on, and it has public code and data. I ran it unchanged.

- **On their own metric** (cumulative return of the single top stock), it reproduces their paper almost exactly: 1.05 vs 1.06. But that metric is very noisy: seeds range from 0.16 to 1.75.
- **On our top-5 Sharpe**, the same predictions score below holding every stock (1.53) under every epoch rule:

| Epoch rule | Sharpe |
|---|---|
| Their rule | 0.45 |
| Our rule | 0.93 |
| Best test epoch | 1.41 |

**Land:** the authors' own model shows the same pattern as ours. That points at the benchmark and the protocol, not our code.

## 20. Step 5: the official STHGCN code

The THINK authors' earlier model can't be run as published:
- **The data is deleted.** Both download links return 404, there is no mirror, and the files needed to rebuild it were never published.
- **The code is broken.** The evaluator crashes on its first call, and the code loads a graph file nothing in the repository creates.
- **The test set is checked every epoch,** with no rule for which epoch to report.

In both earlier codebases from this group, the test set is checked every epoch. That fits option A from step 1. This is an inference: the THINK paper never states its rule, which is why we drafted an email.

## 21. Step 5: 88 readings of the rules

I rescored every model under 88 combinations of:
- how the epoch is chosen
- how prices are scaled
- which Sharpe formula is used
- how many stocks are bought
- which NDCG version is used

Three findings:
- **Sharpe.** Under the formula the paper actually prints (not annualised), everything scores 0.06-0.19, nowhere near 1.18.
- **NDCG.** A correct NDCG gives about 0.57 for every model; random scores 0.564. The paper's 0.86 appears only with a buggy evaluator from the authors' earlier code, under which 43% of random models reach 0.86.
- **Ordering.** In 0 of 88 combinations does THINK > Euclidean > STHGCN hold with both gaps beyond noise.

## 22. Summary of every route

The paper's numbers appear in exactly one place: when the test year picks the epoch. Every fair route shows the same thing: no ranking skill beyond holding the market, and no reliable advantage for THINK over its Euclidean version. That includes:
- fixing our own bug
- variants
- other years
- the authors' code

## 23. Conclusion

We cannot recreate THINK's advantage under any fair evaluation we could build. Its numbers appear only under one of these:
- the test year picks the model
- the Sharpe is annualised
- NDCG has a known bug

The benchmark itself shows no stable daily ranking signal, even for the original authors' code.

Caveat: this is about our reimplementation on their public data. THINK's code is unpublished, so an undisclosed detail could matter. But every route we could build points the same way.

## 24. Limits

| Not run | Why I'd expect the same result | Cost to close |
|---|---|---|
| Market-neutral (long-short) scoring | Removes beta; checks whether any signal hides under market exposure | CPU only |
| 1 day per training step (we used 8) | — | — |
| Hyperparameter search after the fix | Tuning can't create signal the authors' own code lacks | — |
| 25-seed THINK vs Euclidean comparison | It could settle the ordering, but neither model shows skill, so a winner wouldn't mean much | — |
| THINK's own code and protocol | Only the authors can answer | — |
| TSE dataset | Not public | — |

## 25. Call to action

Four decisions today:
1. **Accept the conclusion and pivot.** Stop trying to reproduce THINK on this benchmark.
2. **Send the email to the authors.** The draft is ready in `docs/phase2/EMAIL_AUTHORS.md`. It asks for their epoch rule, Sharpe formula and code.
3. **Pick the pivot direction.** Options:
   - longer prediction horizons (weekly or monthly)
   - richer inputs (volume, fundamentals)
   - a benchmark that actually has a signal
4. **Optional: market-neutral check.** Rescore existing predictions long-short to strip out market beta. CPU only.

Nothing is wasted. These all carry over:
- the leak-free evaluator
- the planted-signal tests
- the random-portfolio comparisons
- the survivor-free data pipeline
- the Kaggle automation

---

## Likely questions

| Question | Answer |
|---|---|
| "Maybe you just implemented it wrong." | Three things argue against it. The planted-signal test shows our pipeline learns real signal (63-84%). The paper-protocol run reproduces the paper's ordering. And the authors' own RSR code shows the same no-skill pattern. |
| "You didn't tune." | Equal-budget tuning in Phase 1 found nothing. And tuning can't create signal that the authors' own model also lacks. |
| "One year is noise." | Agreed, which is why we tested 2018-2023 twice, frozen and retrained, on survivor-free data. |
| "Isn't a Sharpe near 2 great?" | Not in 2017. Holding everything gave 1.53, and random 5-stock picks reach the same range. |
| "Did they cheat?" | We don't claim that. The paper doesn't state its epoch rule. Their earlier code checks the test set every epoch, and that protocol reproduces their numbers. We're asking them directly. |
| "Why not NASDAQ ranking?" | The paper doesn't claim it: Table II ranks only NYSE and TSE, and NASDAQ appears only as classification (which we tested; it failed). TSE data isn't public. |

Evidence for every number: `docs/phase2/REPORT_CLOSURE.md`, which links each phase report.
