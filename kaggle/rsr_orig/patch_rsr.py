"""Apply the MINIMAL patches to the authors' relation_rank_lstm.py (TF1 -> TF2 compat, seed/epoch/output hooks).
Every patch asserts that its target string occurs exactly the expected number of times, and prints itself, so the list is auditable.
Model, loss, optimiser, batch construction, epoch selection and the authors' evaluator are NOT touched.
usage: python patch_rsr.py <training dir>   (patches relation_rank_lstm.py in place; keeps relation_rank_lstm.py.orig)"""
import shutil
import sys
from pathlib import Path

NL = chr(10)
BS = chr(92)   # the source file contains a literal backslash-t inside one print string

d = Path(sys.argv[1])
f = d / "relation_rank_lstm.py"
if not (d / "relation_rank_lstm.py.orig").exists():
    shutil.copy2(f, d / "relation_rank_lstm.py.orig")
s = (d / "relation_rank_lstm.py.orig").read_text()

COMPAT = NL.join([
    "import tensorflow.compat.v1 as tf",
    "tf.disable_v2_behavior()",
    "if not hasattr(getattr(tf, 'layers', None), 'dense'):",
    "    class _Layers(object):",
    "        @staticmethod",
    "        def dense(inputs, units, activation=None, kernel_initializer=None, name=None, **kw):",
    "            with tf.variable_scope(name, default_name='dense'):",
    "                _k = tf.get_variable('kernel', [int(inputs.shape[-1]), units], dtype=tf.float32,",
    "                                     initializer=kernel_initializer if kernel_initializer is not None else tf.glorot_uniform_initializer())",
    "                _b = tf.get_variable('bias', [units], dtype=tf.float32, initializer=tf.zeros_initializer())",
    "                _o = (tf.matmul(inputs, _k) if len(inputs.shape) == 2 else tf.tensordot(inputs, _k, [[len(inputs.shape) - 1], [0]])) + _b",
    "                return _o if activation is None else activation(_o)",
    "    tf.layers = _Layers",
    ""])

TESTLINE = "            print('" + BS + "t Test performance:', cur_test_perf)" + NL
DUMP = NL.join([
    "            _od = os.environ.get('RSR_OUT')",
    "            if _od:",
    "                import json as _json",
    "                os.makedirs(_od, exist_ok=True)",
    "                np.save(os.path.join(_od, 'ep%03d_val_pred.npy' % i), cur_valid_pred.astype(np.float32))",
    "                np.save(os.path.join(_od, 'ep%03d_test_pred.npy' % i), cur_test_pred.astype(np.float32))",
    "                if i == 0:",
    "                    for _n, _a in (('val_gt', cur_valid_gt), ('val_mask', cur_valid_mask), ('test_gt', cur_test_gt), ('test_mask', cur_test_mask)):",
    "                        np.save(os.path.join(_od, _n + '.npy'), _a.astype(np.float32))",
    "                with open(os.path.join(_od, 'epochs.jsonl'), 'a') as _f:",
    "                    _f.write(_json.dumps({'epoch': i, 'val_loss': float(val_loss / (self.test_index - self.valid_index)),",
    "                                          'valid': {k: float(v) for k, v in cur_valid_perf.items()},",
    "                                          'test': {k: float(v) for k, v in cur_test_perf.items()}}) + chr(10))",
    ""])
SAVE = NL + NL.join([
    "    _od = os.environ.get('RSR_OUT')",
    "    if _od:",
    "        for _n, _a in zip(('best_valid_pred', 'best_valid_gt', 'best_valid_mask', 'best_test_pred', 'best_test_gt', 'best_test_mask'), pred_all):",
    "            np.save(os.path.join(_od, _n + '.npy'), np.asarray(_a, dtype=np.float32))",
    ""])

patches = [  # (name, old, new, expected count)
    ("P1 TF2 compat: import tf.compat.v1, disable v2 behaviour, and (only if tf.compat.v1.layers.dense is gone, TF >= 2.16 / Keras 3) "
     "define a drop-in tf.layers.dense (glorot-uniform kernel, zero bias, same maths as TF1 tf.layers.dense)",
     "import tensorflow as tf" + NL, COMPAT, 1),
    ("P2 seed from env RSR_SEED (default = the authors' constant 123456789)", "seed = 123456789",
     "seed = int(os.environ.get('RSR_SEED', '123456789'))", 3),
    ("P3 epochs from env RSR_EPOCHS (default = the authors' 50; used for the 1-epoch smoke test)",
     "steps=1, epochs=50, batch_size=None, gpu=args.gpu,",
     "steps=1, epochs=int(os.environ.get('RSR_EPOCHS', '50')), batch_size=None, gpu=args.gpu,", 1),
    ("P4 per-epoch dump of val/test predictions and the authors' per-epoch performance dicts (read-only hook after their evaluate())",
     TESTLINE, TESTLINE + DUMP, 1),
    ("P5 save the authors' best (val-loss epoch) arrays returned by train()", "    pred_all = RR_LSTM.train()", "    pred_all = RR_LSTM.train()" + SAVE, 1),
]
for name, old, new, n in patches:
    c = s.count(old)
    assert c == n, f"{name}: expected {n} occurrence(s) of {old!r}, found {c}"
    s = s.replace(old, new)
    print("applied:", name)
f.write_text(s)
