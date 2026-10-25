"""Use the skill's PNG renderer with Word's native PDF instead of absent LibreOffice."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import json

root=Path('C:/Users/rsand/.cache/codex-runtimes/codex-primary-runtime/dependencies')
manifest=json.loads((root/'bin/override/native-executables.json').read_text())
bins={str((root/'bin/override'/value).resolve().parent) for value in manifest.values()}
os.environ['PATH']=os.pathsep.join(bins)+os.pathsep+os.environ['PATH']
temp=Path('tmp/benchmark_render_cache');temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp.resolve())
path=Path('C:/Users/rsand/.codex/plugins/cache/openai-primary-runtime/documents/26.1009.11318/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('skill_renderer',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.convert_to_pdf=lambda *args,**kwargs:(str(Path('tmp/benchmark_doc_corrected.pdf').resolve()),'Native Microsoft Word PDF export used; LibreOffice unavailable')
sys.argv=[str(path),'docs/largecap_retraining_20261010/Hypershift_Large_Company_Retraining_Results.docx','--output_dir','tmp/benchmark_doc_after']
module.main()
