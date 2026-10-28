import unittest
import numpy as np
from rishi_source_device_trace import array_digest, batch_for_origin


class SourceDeviceTraceTests(unittest.TestCase):
    def test_input_digest_includes_shape_dtype_and_order(self):
        x=np.arange(8,dtype=np.float32)
        self.assertNotEqual(array_digest(x),array_digest(x.reshape(2,4)))
        self.assertNotEqual(array_digest(x),array_digest(x.astype(np.float64)))
        self.assertNotEqual(array_digest(x),array_digest(x[::-1]))

    def test_batch_matches_original_array_split_and_rejects_missing(self):
        ids=np.arange(20);np.testing.assert_array_equal(batch_for_origin(ids,8,9),np.arange(7,14))
        with self.assertRaises(ValueError):batch_for_origin(ids,8,99)
