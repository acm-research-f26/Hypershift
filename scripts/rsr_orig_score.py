"""Score the authors' ORIGINAL RSR-I (NYSE) predictions, saved by kaggle/rsr_orig/run_rsr_orig.py, with OUR evaluator.
usage: CUDA_VISIBLE_DEVICES=-1 python scripts/rsr_orig_score.py <out dir holding seed_*/> [--ours results/R8_baselines_g2/RSR_I]
Their test column c is target day 1008 + c (offset 1000 + c, seq 8, steps 1) = our test day c, 237 days: same indexing as
hypershift.train.loop.window_offsets for split 'test'. gt/mask are compared with our own saved test_gt/test_mask; metrics use OUR gt and mask (primary), 'sr_their_gt_mask' uses theirs.
Epoch selections: THEIR rule (min mean validation LOSS = reg + alpha*rank, strict <, first epoch wins ties; code of relation_rank_lstm.py),
OURS (max validation Sharpe) and ORACLE (max test Sharpe, the paper-style protocol; diagnostic)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from hypershift.eval.metrics import evaluate_all, sharpe, topk_daily_returns


def ic_daily(pred, gt, mask):
    v = []
    for d in range(pred.shape[1]):
        i = np.nonzero(mask[:, d] > 0.5)[0]
        if len(i) > 2 and np.std(pred[i, d]) > 0:
            v.append(spearmanr(pred[i, d], gt[i, d])[0])
    return float(np.mean(v)), len(v)


def sr_random_tiebreak(pred, gt, mask, draws=20, k=5, seed=0):
    """Sharpe of the daily top-k with TIES BROKEN AT RANDOM (our evaluator's argsort(kind='stable') breaks ties by ticker index)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        r = np.zeros(pred.shape[1])
        for d in range(pred.shape[1]):
            i = np.nonzero(mask[:, d] > 0.5)[0]
            o = np.lexsort((rng.random(len(i)), -pred[i, d]))[:k]
            r[d] = gt[i[o], d].mean()
        out.append(sharpe(r))
    return float(np.mean(out)), float(np.std(out))


def hold_all(gt, mask):
    return np.array([gt[mask[:, d] > 0.5, d].mean() for d in range(gt.shape[1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ours", default="results/R8_baselines_g2/RSR_I")
    a = ap.parse_args()
    res = {}
    ours_gt = np.load(Path(a.ours) / "seed_0/test_gt.npy")
    ours_mask = np.load(Path(a.ours) / "seed_0/test_mask.npy")
    for sd in sorted(Path(a.out).glob("seed_*")):
        if not (sd / "DONE").exists():
            continue
        tp = np.load(sd / "test_pred_all.npy").astype(np.float64)
        vp = np.load(sd / "val_pred_all.npy").astype(np.float64)
        tg, tm = np.load(sd / "test_gt.npy").astype(np.float64), np.load(sd / "test_mask.npy").astype(np.float64)
        vg, vm = np.load(sd / "val_gt.npy").astype(np.float64), np.load(sd / "val_mask.npy").astype(np.float64)
        ep = [json.loads(l) for l in open(sd / "epochs.jsonl")]
        E = len(ep)
        assert tp.shape[0] == E and tp.shape[2] == 237 and tg.shape == ours_gt.shape, (tp.shape, tg.shape, ours_gt.shape)
        both = (tm > .5) & (ours_mask > .5)
        gt_diff = float(np.abs(tg - ours_gt)[both].max())
        mask_equal = bool(np.array_equal(tm, ours_mask))
        val_loss = np.array([e["val_loss"] for e in ep])
        best, e_their = np.inf, 0
        for i, v in enumerate(val_loss):
            if v < best:
                best, e_their = v, i
        val_sr = np.array([sharpe(topk_daily_returns(vp[i], vg, vm, 5)) for i in range(E)])
        test_sr = np.array([sharpe(topk_daily_returns(tp[i], tg, tm, 5)) for i in range(E)])
        e_ours, e_orc = int(np.argmax(val_sr)), int(np.argmax(test_sr))
        row = {"epochs": E, "gt_maxabsdiff_vs_ours_on_common_mask": gt_diff, "mask_equal_ours": mask_equal, "n_masked_pairs": int(tm.sum()),
               "epoch_their_rule": e_their, "epoch_our_rule": e_ours, "epoch_oracle": e_orc}
        for tag, e in (("their_rule", e_their), ("our_rule", e_ours), ("oracle", e_orc), ("last", E - 1)):
            m = evaluate_all(tp[e], ours_gt, ours_mask)   # PRIMARY: our gt and mask (same as the R8 runs)
            m["sr_their_gt_mask"] = sharpe(topk_daily_returns(tp[e], tg, tm, 5))
            ic, nd = ic_daily(tp[e], ours_gt, ours_mask)
            m["sr_random_tiebreak_mean"], m["sr_random_tiebreak_sd"] = sr_random_tiebreak(tp[e], ours_gt, ours_mask)
            m["n_tied_values_per_day"] = float(np.mean([len(np.unique(tp[e][ours_mask[:, d] > .5, d])) for d in range(237)]))
            m.update(ic=ic, ic_days=nd, val_sr=float(val_sr[e]), their_printed=ep[e]["test"], their_printed_valid=ep[e]["valid"],
                     distinct_top5_sets=len({tuple(sorted(np.argsort(-np.where(ours_mask[:, d] > .5, tp[e][:, d], -np.inf), kind="stable")[:5])) for d in range(237)}),
                     pred_std=float(np.std(tp[e][ours_mask > .5])))
            row[tag] = m
        h = hold_all(ours_gt, ours_mask)
        row["hold_all_sr"] = sharpe(h)
        row["hold_all_sr_their_gt_mask"] = sharpe(hold_all(tg, tm))
        row["theirs_only_mask_entries"] = int(((tm > .5) & (ours_mask < .5)).sum())
        row["ours_only_mask_entries"] = int(((tm < .5) & (ours_mask > .5)).sum())
        res[sd.name] = row
    json.dump(res, open(Path(a.out) / "scores.json", "w"), indent=1)
    for k, r in res.items():
        print(k, {t: (round(r[t]["sr"], 3), round(r[t]["ndcg5"], 4), round(r[t]["ic"], 4)) for t in ("their_rule", "our_rule", "oracle")},
              "epochs", r["epoch_their_rule"], r["epoch_our_rule"], r["epoch_oracle"], "gt diff", r["gt_maxabsdiff_vs_ours_on_common_mask"], "mask eq", r["mask_equal_ours"], r["theirs_only_mask_entries"], r["ours_only_mask_entries"],
              "hold-all", round(r["hold_all_sr"], 3))


if __name__ == "__main__":
    main()
