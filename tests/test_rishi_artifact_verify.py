import hashlib
from pathlib import Path
import tempfile
import unittest
from rishi_artifact_verify import verify_contract,write_receipt


class ArtifactVerifyTests(unittest.TestCase):
    def test_ready_requires_every_original_hash(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            p=Path(folder);(p/'runs').mkdir();(p/'runs/a.bin').write_bytes(b'original')
            c={'files':[{'path':'runs/a.bin','sha256':hashlib.sha256(b'original').hexdigest()}]}
            r=verify_contract(c,p)
            self.assertEqual(r['status'],'hashes_ready_not_raw_replayed')
            self.assertEqual(r['matched'],1)

    def test_missing_and_mutated_originals_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            p=Path(folder);(p/'a.bin').write_bytes(b'changed')
            sha=hashlib.sha256(b'original').hexdigest()
            c={'files':[{'path':'a.bin','sha256':sha},{'path':'absent.bin','sha256':sha}]}
            r=verify_contract(c,p)
            self.assertEqual(r['status'],'not_ready')
            self.assertEqual((r['matched'],r['missing'],r['mismatched']),(0,1,1))

    def test_unsafe_or_case_colliding_contract_is_rejected(self):
        sha='a'*64
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            for names in [['../outside.bin'],['C:/outside.bin'],['runs/A.bin','runs/a.bin'],
                          ['runs/a.bin','runs//a.bin'],['runs/a.bin','runs/./a.bin']]:
                with self.subTest(names=names):
                    with self.assertRaises(ValueError):
                        verify_contract({'files':[{'path':n,'sha256':sha} for n in names]},Path(folder))

    def test_empty_contract_never_implies_ready(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            with self.assertRaises(ValueError):verify_contract({'files':[]},Path(folder))

    def test_existing_receipt_survives_even_when_prior_exists_check_is_stale(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            path=Path(folder)/'receipt.json'
            path.write_text('earlier evidence')
            with self.assertRaises(FileExistsError):write_receipt(path,{'status':'different'})
            self.assertEqual(path.read_text(),'earlier evidence')


if __name__=='__main__':unittest.main()
