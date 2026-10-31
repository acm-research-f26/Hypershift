"""Independent matched-initialization/RNG checks and actual resumed computation."""
from datetime import datetime, timezone
import json
import shutil

import numpy as np
import torch
import run_ndx_consistency_retraining as study


def audit():
    audit_plan = json.loads((study.RUN/'TRAINING_AUDIT_PROTOCOL.json').read_text(encoding='utf-8-sig'))
    assert study.sha(study.ROOT/'research/evaluation/audit_ndx_consistency_retraining.py') == audit_plan['source_sha256']
    assert study.sha(study.RUN/'PROTOCOL.json') == audit_plan['experiment_protocol_sha256']
    plan = study.verify()
    lock = json.loads((study.RUN/'FITS_COMPLETE.json').read_text())
    assert lock['fits'] == 320
    for name, expected in lock['hashes'].items():
        assert study.sha(study.RUN/name) == expected, name
    torch.set_num_threads(2)
    checks = []
    for family in plan['families']:
        for year in plan['years']:
            for seed in plan['seeds']:
                a = study.RUN/'fits/dropout_control'/family/str(year)/str(seed)
                b = study.RUN/'fits/consistency'/family/str(year)/str(seed)
                first = torch.load(a/'epoch_00.pt', map_location='cpu', weights_only=False)
                second = torch.load(b/'epoch_00.pt', map_location='cpu', weights_only=False)
                error = max(float((first['state_dict'][k].double()-second['state_dict'][k].double()).abs().max())
                            for k in first['state_dict'])
                assert error == 0
                for epoch in range(4):
                    aa = torch.load(a/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
                    bb = torch.load(b/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
                    assert torch.equal(aa['torch_rng'], bb['torch_rng'])
                    assert torch.equal(aa['cuda_rng'], bb['cuda_rng'])
                parity = [json.loads((p/'INITIAL_PARITY.json').read_text())['validation_max_abs_error'] for p in (a,b)]
                checks.append(dict(family=family, year=year, seed=seed, initial_parameter_max_error=error,
                                   rng_equal_every_epoch=True, source_forecast_max_error=max(parity)))
    original_run = study.RUN
    replay_results = []
    device = study.configure_device('cuda')
    for family in plan['families']:
        year, seed, arm = 2025, 7, 'consistency'
        source = original_run/'fits'/arm/family/str(year)/str(seed)
        run = original_run/'resume_checks'/family
        dest = run/'fits'/arm/family/str(year)/str(seed)
        if dest.exists():
            raise FileExistsError('Preserve completed resume diagnostic: '+str(dest))
        dest.mkdir(parents=True); (run/'worker_progress').mkdir()
        for name in ('epoch_00.pt','epoch_01.pt','validation_00.npz','validation_01.npz','INITIAL_PARITY.json'):
            shutil.copyfile(source/name, dest/name)
        prepared = study.trainer.prepare(year, device)
        try:
            study.RUN = run
            study.fit(prepared, family, year, seed, arm, plan, device)
        finally:
            study.RUN = original_run
        maximum = 0.
        ema_error = 0.
        for epoch in (2,3):
            a = torch.load(source/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
            b = torch.load(dest/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
            maximum = max(maximum, max(float((a['state_dict'][k].double()-b['state_dict'][k].double()).abs().max()) for k in a['state_dict']))
            ema_error = max(ema_error, max(float((a['ema']['shadow'][k].double()-b['ema']['shadow'][k].double()).abs().max()) for k in a['ema']['shadow']))
            assert torch.equal(a['torch_rng'], b['torch_rng']) and torch.equal(a['cuda_rng'], b['cuda_rng'])
        forecast_error = 0.
        for method in ('raw','ema'):
            with np.load(source/f'{method}_predictions.npz') as a, np.load(dest/f'{method}_predictions.npz') as b:
                forecast_error = max(forecast_error, *(float(np.max(np.abs(a[k]-b[k]))) for k in a.files))
        assert maximum == ema_error == forecast_error == 0
        replay_results.append(dict(family=family, year=year, seed=seed, resumed_epochs=[2,3],
            parameter_error=maximum, ema_error=ema_error, forecast_error=forecast_error))
        del prepared; torch.cuda.empty_cache()
    result = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        paired_fits=len(checks), scientific_fits=320, additional_actual_resume_diagnostics=2,
        all_initial_parameters_exactly_matched=True, all_cpu_cuda_rng_states_equal_every_epoch=True,
        source_forecast_max_error=max(r['source_forecast_max_error'] for r in checks),
        checks=checks, actual_resumes=replay_results,
        source_sha256=study.sha(study.ROOT/'research/evaluation/audit_ndx_consistency_retraining.py'),
        fit_lock_sha256=study.sha(original_run/'FITS_COMPLETE.json'))
    study.write(original_run/'TRAINING_AUDIT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}, indent=2), flush=True)


if __name__ == '__main__': audit()
