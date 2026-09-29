"""Standard NDCG@5 vs the NDCG computed by the authors' STHAN-SR evaluator (training/evaluator.py lines 19-43,
https://github.com/NDS-VU/STHAN-SR-AAAI21, commit 8d7861c). Run from repo root."""
import numpy as np
from sklearn.metrics import ndcg_score
from hypershift.data.rsr import load_rsr
from hypershift.eval.baselines import baseline_scores


def sthan_ndcg(pred, gt, mask):
    """Verbatim logic: sets of top-5 *stock indices* passed as relevance/score; overwritten every day -> last day only."""
    for i in range(pred.shape[1]):
        rank_gt, gt_top5 = np.argsort(gt[:, i]), set()
        for j in range(1, pred.shape[0] + 1):
            c = rank_gt[-j]
            if mask[c][i] >= 0.5 and len(gt_top5) < 5:
                gt_top5.add(c)
        rank_pre, pre_top5 = np.argsort(pred[:, i]), set()
        for j in range(1, pred.shape[0] + 1):
            c = rank_pre[-j]
            if mask[c][i] >= 0.5 and len(pre_top5) < 5:
                pre_top5.add(c)
        out = ndcg_score(np.array(list(gt_top5)).reshape(1, -1), np.array(list(pre_top5)).reshape(1, -1))
    return out


def std_ndcg(pred, gt, mask, k=5):
    """Standard: relevance = true return (shifted >= 0), averaged over all test days."""
    v = []
    for i in range(pred.shape[1]):
        idx = np.nonzero(mask[:, i] > 0.5)[0]
        v.append(ndcg_score((gt[idx, i] - gt[idx, i].min())[None], pred[idx, i][None], k=k))
    return float(np.mean(v))


if __name__ == "__main__":
    gt = np.array([[.05], [.03], [.01], [-.02], [0.], [.04]]); m = np.ones_like(gt)
    for name, p in [("perfect", gt), ("reversed (worst)", -gt)]:
        print(f"toy {name:18s} standard {std_ndcg(p, gt, m):.3f} | STHAN-code {sthan_ndcg(p, gt, m):.3f}")
    d = load_rsr("data/raw/rsr/data", "NYSE", "paper")
    p, g, mk = baseline_scores(d, 16, "test", "oracle")
    for name, q in [("oracle", p), ("inverse", -p), ("random", baseline_scores(d, 16, "test", "random")[0])]:
        print(f"NYSE {name:8s} standard {std_ndcg(q, g, mk):.3f} | STHAN-code {sthan_ndcg(q, g, mk):.3f}")
    rs = np.array([sthan_ndcg(baseline_scores(d, 16, "test", "random", seed=s)[0], g, mk) for s in range(200)])
    print(f"STHAN-code NDCG, 200 random models: mean {rs.mean():.3f}, 5-95% [{np.percentile(rs, 5):.3f}, {np.percentile(rs, 95):.3f}], share >= 0.86: {(rs >= 0.86).mean():.2f}")
