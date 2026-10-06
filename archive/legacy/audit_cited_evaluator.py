"""Demonstrate the stock-ID NDCG problem in the cited baseline's expression.

Does not import/execute the downloaded evaluator; reproduces only its supplied
expression with a separately implemented standard linear-gain NDCG calculation.
This cannot establish which evaluator THINK actually used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from run_think_reconstruction import dcg,metrics


def id_ndcg(pred,truth,k=5):
    truth_ids=set(np.argsort(truth)[-k:].tolist())
    predicted_ids=set(np.argsort(pred)[-k:].tolist())
    # Source passes list(gt_top5) as y_true and list(pre_top5) as y_score.
    gains=np.asarray(list(truth_ids),dtype=float)
    scores=np.asarray(list(predicted_ids),dtype=float)
    return dcg(gains,scores,k)/dcg(gains,gains,k)


def main():
    rng=np.random.default_rng(17)
    truth=rng.normal(size=64); pred=rng.normal(size=64)
    for _ in range(100):
        perm=rng.permutation(64)
        before,after=id_ndcg(pred,truth),id_ndcg(pred[perm],truth[perm])
        if abs(before-after)>1e-4: break
    assert abs(before-after)>1e-4
    correct_before=metrics(pred[None],truth[None],np.ones((1,64)))['ndcg_at_5_positive_return']
    correct_after=metrics(pred[perm][None],truth[perm][None],np.ones((1,64)))['ndcg_at_5_positive_return']
    assert np.isclose(correct_before,correct_after)
    source=Path('data/think_reproduction_sources/evaluator.py.txt')
    output={'scope':'Cited STHAN-SR evaluator expression; not proof of THINK evaluation behavior.',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'actual_returns':truth.tolist(),'predicted_scores':pred.tolist(),'new_node_order':perm.tolist(),
            'source_expression_before_relabeling':before,'source_expression_after_relabeling':after,
            'correct_ndcg_before_relabeling':correct_before,'correct_ndcg_after_relabeling':correct_after,
            'meaning':'Only stock numbering changed. Correct ranking quality is invariant; the source expression is not.'}
    Path('data/think_reproduction_sources/evaluator_counterexample.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k not in ['actual_returns','predicted_scores','new_node_order']},indent=2))


if __name__=='__main__': main()
