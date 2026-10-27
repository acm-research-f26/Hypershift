"""Filesystem and subprocess checks for the research compatibility launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hypershift.workspace import ROOT,CompatibilityWorkspace,catalog,clean,digest,process_alive,run_python,runtime_sources


class WorkspaceLayoutTests(unittest.TestCase):
    def setUp(self):
        self.base=ROOT/'.cache/layout_tests';self.base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.base);self.root=Path(self.temp.name)
        self.sources=self.root/'research/core';self.sources.mkdir(parents=True)

    def tearDown(self):
        self.assertEqual(self.root.resolve().parent,self.base.resolve())
        self.temp.cleanup()

    def source(self,name,content):
        p=self.sources/name;p.write_bytes(content);return p

    def test_original_imports_file_paths_and_hashes_are_preserved(self):
        dependency=self.source('dependency.py',b'VALUE=17\n')
        script=self.source('example.py',b'from pathlib import Path\nimport dependency\nprint(dependency.VALUE)\nprint(Path(__file__).resolve().parent)\n')
        with CompatibilityWorkspace(self.root) as workspace:
            self.assertEqual(digest(self.root/'example.py'),digest(script))
            result=subprocess.run([sys.executable,str(self.root/'example.py')],cwd=self.root,text=True,capture_output=True,check=True)
            self.assertEqual(result.stdout.splitlines(),['17',str(self.root)])
        self.assertFalse((self.root/'example.py').exists());self.assertFalse((self.root/'dependency.py').exists())
        self.assertTrue(dependency.exists());self.assertTrue(script.exists())

    def test_existing_root_file_is_never_overwritten_or_removed(self):
        self.source('a.py',b'A=1\n');self.source('b.py',b'B=2\n')
        existing=self.root/'b.py';existing.write_text('user edits')
        with self.assertRaisesRegex(RuntimeError,'overwrite'):
            with CompatibilityWorkspace(self.root):pass
        self.assertEqual(existing.read_text(),'user edits');self.assertFalse((self.root/'a.py').exists())

    def test_nonzero_child_exit_cleans_sources_and_preserves_generated_output(self):
        self.source('failure.py',b'from pathlib import Path\nPath("result.txt").write_text("keep")\nraise SystemExit(7)\n')
        self.assertEqual(run_python([str(self.root/'failure.py')],self.root),7)
        self.assertFalse((self.root/'failure.py').exists());self.assertEqual((self.root/'result.txt').read_text(),'keep')

    def test_user_edit_to_temporary_copy_is_preserved(self):
        original=self.source('example.py',b'X=1\n')
        with self.assertRaisesRegex(RuntimeError,'edited'):
            with CompatibilityWorkspace(self.root):(self.root/'example.py').write_text('edited')
        self.assertEqual((self.root/'example.py').read_text(),'edited');self.assertEqual(original.read_bytes(),b'X=1\n')

    def test_edits_to_organized_source_survive_cleanup(self):
        original=self.source('example.py',b'X=1\n')
        with CompatibilityWorkspace(self.root):original.write_bytes(b'X=2\n')
        self.assertEqual(original.read_bytes(),b'X=2\n');self.assertFalse((self.root/'example.py').exists())

    def test_duplicate_module_names_are_rejected(self):
        self.source('example.py',b'X=1\n');other=self.root/'research/models';other.mkdir()
        (other/'example.py').write_bytes(b'X=2\n')
        with self.assertRaisesRegex(ValueError,'Duplicate'):catalog(self.root)

    def test_lease_cannot_escape_workspace(self):
        state=self.root/'.cache/source_layout_runtime';state.mkdir(parents=True)
        (state/'lease.json').write_text(json.dumps(dict(owner_pid=0,child_pid=0,aliases={'../outside.py':'fake'})))
        with self.assertRaisesRegex(RuntimeError,'Unsafe'):clean(self.root)

    def test_stale_unchanged_view_can_be_cleaned(self):
        data=b'X=1\n';p=self.root/'example.py';p.write_bytes(data)
        state=self.root/'.cache/source_layout_runtime';state.mkdir(parents=True)
        (state/'lease.json').write_text(json.dumps(dict(owner_pid=0,child_pid=0,aliases={'example.py':digest(p)})))
        clean(self.root);self.assertFalse(p.exists());self.assertFalse((state/'lease.json').exists())

    def test_running_process_lease_is_not_cleaned(self):
        p=self.root/'example.py';p.write_bytes(b'X=1\n')
        state=self.root/'.cache/source_layout_runtime';state.mkdir(parents=True)
        (state/'lease.json').write_text(json.dumps(dict(owner_pid=os.getpid(),child_pid=None,aliases={'example.py':digest(p)})))
        with self.assertRaisesRegex(RuntimeError,'still running'):clean(self.root)
        self.assertTrue(p.exists());self.assertTrue(process_alive(os.getpid()))

    def test_nested_launches_are_rejected(self):
        self.source('example.py',b'X=1\n')
        with CompatibilityWorkspace(self.root):
            with self.assertRaisesRegex(RuntimeError,'owns the compatibility view'):
                with CompatibilityWorkspace(self.root):pass
        self.assertFalse((self.root/'example.py').exists())

    def test_pinned_requirements_have_their_original_runtime_names(self):
        folder=self.root/'requirements';folder.mkdir()
        p=folder/'requirements-cuda.txt';p.write_bytes(b'torch==2.14.0+cu130\n')
        with CompatibilityWorkspace(self.root):self.assertEqual((self.root/p.name).read_bytes(),p.read_bytes())
        self.assertFalse((self.root/p.name).exists())

    def test_latest_sources_are_discovered_without_archived_sources(self):
        latest=self.root/'latest/models';latest.mkdir(parents=True)
        (latest/'current.py').write_bytes(b'X=1\n')
        archive=self.root/'archive/studies_20261025/research/models';archive.mkdir(parents=True)
        (archive/'earlier.py').write_bytes(b'X=2\n')
        self.assertIn('current.py',catalog(self.root))
        self.assertNotIn('earlier.py',catalog(self.root))
        self.assertIn('earlier.py',catalog(self.root,include_archived=True))

    def test_latest_protocol_uses_exact_original_path_then_cleans_up(self):
        config=self.root/'config';config.mkdir()
        latest=self.root/'latest/config';latest.mkdir(parents=True)
        canonical=latest/'study.json';canonical.write_bytes(b'{"study":"frozen"}\n')
        record={'files':[{'path':'latest/config/study.json','runtime_alias':'config/study.json','role':'latest'}]}
        (config/'study_layout_20261025.json').write_text(json.dumps(record))
        with CompatibilityWorkspace(self.root):
            self.assertEqual((config/'study.json').read_bytes(),canonical.read_bytes())
        self.assertFalse((config/'study.json').exists());self.assertTrue(canonical.exists())

    def test_archived_protocol_aliases_require_explicit_opt_in(self):
        config=self.root/'config';config.mkdir()
        archived=self.root/'archive/studies_20261025/config';archived.mkdir(parents=True)
        (archived/'old.json').write_bytes(b'{}\n')
        record={'files':[{'path':'archive/studies_20261025/config/old.json','runtime_alias':'config/old.json','role':'archive'}]}
        (config/'study_layout_20261025.json').write_text(json.dumps(record))
        self.assertNotIn('config/old.json',runtime_sources(self.root))
        self.assertIn('config/old.json',runtime_sources(self.root,include_archived=True))

    def test_nested_alias_escape_is_rejected_before_copying(self):
        source=self.source('example.py',b'X=1\n')
        with self.assertRaisesRegex(RuntimeError,'Unsafe'):
            with CompatibilityWorkspace(self.root,{'config/../../outside.py':source}):pass
        self.assertFalse((self.root.parent/'outside.py').exists())


if __name__=='__main__':unittest.main()
