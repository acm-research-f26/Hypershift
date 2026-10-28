"""Individual-output input Jacobians for fixed synthetic membership cases."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rishi_member_count import toy_input
from rishi_synthetic_gate import make_model
from rishi_synthetic_robustness import forecast


def individual_jacobian(model, raw, mask, arm):
    x = torch.tensor(raw, requires_grad=True)
    eligible = torch.tensor(mask)
    score = forecast(model, x, eligible, arm)
    jac = torch.stack([torch.autograd.grad(score[i], x, retain_graph=True)[0]
                       for i in np.flatnonzero(mask)])
    return score.detach().numpy(), jac.detach().numpy()


def main():
    doc = Path('docs/rishi_mechanism_20261027')
    out = doc / 'synthetic_mask_jacobian_20261028'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows = []
    for seed in (7, 19, 42):
        model = make_model(seed).eval()
        for case in (0, 1, 2):
            for population in ('diverse', 'replicated_peer'):
                for count in (2, 4, 6):
                    for arm in ('full', 'temporal_only'):
                        results = {}
                        for variant in ('zero', 'nan', 'garbage'):
                            raw, mask = toy_input(case, count, population, variant)
                            results[variant] = individual_jacobian(model, raw, mask, arm)
                        base, jac = results['zero']
                        rows.append(dict(seed=seed, case=case, population=population,
                            active_count=count, arm=arm, output_checks=count,
                            finite=all(np.isfinite(s[mask]).all() and np.isfinite(j).all() for s,j in results.values()),
                            output_equal=all(np.array_equal(results[v][0][mask], base[mask]) for v in ('nan','garbage')),
                            jacobian_equal=all(np.array_equal(results[v][1], jac) for v in ('nan','garbage')),
                            inactive_max=max(float(np.max(np.abs(j[:,~mask]))) for _,j in results.values()),
                            jacobian_max_difference=max(float(np.max(np.abs(results[v][1]-jac))) for v in ('nan','garbage'))))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'CELLS.csv', index=False)
    paths = [Path(__file__), doc/'JACOBIAN_TOY_PROTOCOL.md', doc/'MEMBER_COUNT_TOY_PROTOCOL.md',
             Path('research/analysis/rishi_member_count.py'), Path('research/analysis/rishi_synthetic_gate.py'),
             Path('research/analysis/rishi_synthetic_robustness.py'), doc/'source_snapshot/model.py',
             doc/'source_snapshot/layer.py', Path('latest/models/ndx_reference_models.py')]
    summary = dict(cells=len(frame), individual_outputs=int(frame.output_checks.sum()),
                   finite_failures=int((~frame.finite).sum()), output_failures=int((~frame.output_equal).sum()),
                   jacobian_failures=int((~frame.jacobian_equal).sum()),
                   inactive_max=float(frame.inactive_max.max()),
                   input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'RESULTS.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
