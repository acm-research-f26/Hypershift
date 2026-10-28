import json,tempfile,unittest
from pathlib import Path
import numpy as np
from rishi_compare_device_traces import compare


class CompareDeviceTracesTests(unittest.TestCase):
    def test_matching_traces_and_mismatched_input_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            meta={'all_input_hashes':{'x':'hash'},'batch_ids':[1],'file_sha256':{'7.pt':'abc'},
                'code_modules_sha256':{'model':'def'},'layers':[{'key':'layer_0000','module':'x','call':0,'shape':[1],'dtype':'float32'}],
                'environment':{},'forecast_max_error_pp':{},'batch_vs_full_max_error':0}
            for path in (a,b):
                (path/'TRACE.json').write_text(json.dumps(meta))
                np.savez(path/'ACTIVATIONS.npz',layer_0000=np.array([1.],dtype='float32'))
                np.savez(path/'FORECASTS.npz',validation=np.array([1.]),test=np.array([1.]),batch=np.array([1.]),
                    validation_ids=np.array([1]),test_ids=np.array([1]),batch_ids=np.array([1]),security_ids=np.array(['A']))
            self.assertEqual(compare(a,b)['forecasts']['test'],0)
            meta['all_input_hashes']['x']='other';(b/'TRACE.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):compare(a,b)
            meta['all_input_hashes']['x']='hash';meta['code_modules_sha256']['model']='changed';(b/'TRACE.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):compare(a,b)
