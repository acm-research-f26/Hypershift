"""Compare two fixed device traces without reading training data or credentials."""
import argparse,json
from pathlib import Path
import numpy as np


def compare(local,remote):
    local=Path(local);remote=Path(remote)
    left=json.loads((local/'TRACE.json').read_text());right=json.loads((remote/'TRACE.json').read_text())
    if left['all_input_hashes']!=right['all_input_hashes'] or left['batch_ids']!=right['batch_ids']:
        raise ValueError('Input/axis hashes or canonical batch differ')
    def names(d):return {Path(k.replace('\\','/')).name:v for k,v in d.items()}
    if names(left['file_sha256'])!=names(right['file_sha256']):raise ValueError('Checkpoint, export or trace-code hash differs')
    if not left.get('code_modules_sha256') or left['code_modules_sha256']!=right.get('code_modules_sha256'):raise ValueError('Model or preparation code hashes differ or are missing')
    if left['layers']!=right['layers']:raise ValueError('Layer names, call order, shapes or dtypes differ')
    result={'local_environment':left['environment'],'candidate_environment':right['environment'],
        'local_saved_errors_pp':left['forecast_max_error_pp'],'candidate_saved_errors_pp':right['forecast_max_error_pp'],
        'local_batch_full_error':left['batch_vs_full_max_error'],'candidate_batch_full_error':right['batch_vs_full_max_error'],
        'layers':[],'forecasts':{},'limit':'One checkpoint only; source GPU match does not pass the full replay gate'}
    with np.load(local/'ACTIVATIONS.npz',allow_pickle=False) as a,np.load(remote/'ACTIVATIONS.npz',allow_pickle=False) as b:
        for item in left['layers']:
            x=a[item['key']];y=b[item['key']]
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite or mismatched activation')
            result['layers'].append(dict(module=item['module'],call=item['call'],shape=item['shape'],max_abs=float(np.max(np.abs(x.astype('float64')-y.astype('float64'))))))
    with np.load(local/'FORECASTS.npz',allow_pickle=False) as a,np.load(remote/'FORECASTS.npz',allow_pickle=False) as b:
        for axis in ('validation_ids','test_ids','batch_ids','security_ids'):
            if not np.array_equal(a[axis],b[axis]):raise ValueError('Forecast axis mismatch')
        for split in ('validation','test','batch'):
            if a[split].shape!=b[split].shape or not np.isfinite(a[split]).all() or not np.isfinite(b[split]).all():raise ValueError('Invalid forecast arrays')
            result['forecasts'][split]=float(np.max(np.abs(a[split].astype('float64')-b[split].astype('float64'))))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Choose a fresh comparison file')
    result=compare(args.local,args.candidate);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['forecasts'],indent=2))


if __name__=='__main__':main()
