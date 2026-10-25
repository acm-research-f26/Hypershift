"""Keep historical research paths real, temporary and byte-faithful.

Older experiments hash root-level sources, import their original module names,
and derive data locations from __file__. The compatibility view uses ordinary
file copies: no patched imports, forged hashes, symlink resolution or lock edits.
Only files owned by a recorded lease are removed after the child exits.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
PINNED_REQUIREMENTS=(
    'requirements-cuda.txt','requirements-data.txt','requirements-research-20261020.txt',
    'requirements-tested-cpu.txt','requirements-tested.txt',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog(root=ROOT,include_archived=False):
    """Find grouped modules by unique original basename; include new scripts."""
    root=Path(root).resolve();result={}
    sources=sorted((root/'research').rglob('*.py'))+sorted((root/'tests').rglob('test_*.py'))
    sources+=sorted((root/'latest').rglob('*.py'))
    if include_archived:
        archive=root/'archive/studies_20261025'
        sources+=sorted((archive/'research').rglob('*.py'))+sorted((archive/'tests').glob('test_*.py'))
    for path in sources:
        if path.name=='__init__.py' or '__pycache__' in path.parts:continue
        if path.name in result:raise ValueError(f'Duplicate module basename: {path.name}')
        result[path.name]=path
    for name in PINNED_REQUIREMENTS:
        path=root/'requirements'/name
        if path.exists():result[name]=path
    return result


def runtime_sources(root=ROOT,include_archived=False):
    """Supply relocated protocols at the real paths used by frozen experiments."""
    root=Path(root).resolve();result=catalog(root,include_archived)
    manifest=root/'config/study_layout_20261025.json'
    if manifest.exists():
        for record in json.loads(manifest.read_text(encoding='utf8'))['files']:
            name=record.get('runtime_alias')
            if name and (record['role']=='latest' or include_archived):
                if name in result:raise ValueError('Duplicate runtime alias: '+name)
                result[name]=root/record['path']
    return result


def safe_target(root,name):
    relative=Path(name);target=root/relative
    if relative.is_absolute() or '..' in relative.parts or not target.resolve().is_relative_to(root):
        raise RuntimeError('Unsafe compatibility lease path: '+name)
    if len(relative.parts)>1 and (len(relative.parts)!=2 or relative.parts[0]!='config'):
        raise RuntimeError('Unsafe compatibility lease path: '+name)
    return target


def process_alive(pid):
    """Check a lease owner without sending a signal on Windows."""
    if not pid or pid<=0:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x100000,False,pid)  # SYNCHRONIZE only
        if not handle:return ctypes.get_last_error()!=87  # absent PID
        try:return kernel.WaitForSingleObject(handle,0)==0x102
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


@contextmanager
def command_lock(path):
    """Serialize compatibility views, including launches from other terminals."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:raise RuntimeError('Another Hypershift command owns the compatibility view.') from exc
            try:yield
            finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as exc:raise RuntimeError('Another Hypershift command owns the compatibility view.') from exc
            try:yield
            finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


class CompatibilityWorkspace:
    def __init__(self,root=ROOT,sources=None):
        self.root=Path(root).resolve();self.sources=runtime_sources(self.root) if sources is None else sources
        self.state=self.root/'.cache/source_layout_runtime';self.lease=self.state/'lease.json'
        self.lock=None;self.record=None;self.keep=False

    def _write(self):
        temporary=self.lease.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record,indent=2),encoding='utf8');temporary.replace(self.lease)

    def _cleanup(self,record):
        # Resolve and check every recorded file before removing any of them.
        targets=[]
        for name,expected in record.get('aliases',{}).items():
            path=safe_target(self.root,name)
            if not path.exists():continue
            if path.is_symlink() or not path.is_file() or digest(path)!=expected:
                raise RuntimeError('Compatibility file was edited; preserved for recovery: '+str(path))
            targets.append(path)
        for path in targets:path.unlink()
        if self.lease.exists():self.lease.unlink()

    def _recover(self):
        if not self.lease.exists():return
        prior=json.loads(self.lease.read_text())
        if process_alive(prior.get('owner_pid')) or process_alive(prior.get('child_pid')):
            raise RuntimeError('A previous research command is still running; its source files are preserved.')
        self._cleanup(prior)

    def __enter__(self):
        self.lock=command_lock(self.state/'command.lock');self.lock.__enter__()
        try:
            self._recover();aliases={}
            for name,source in self.sources.items():
                source=Path(source).resolve();target=safe_target(self.root,name)
                if not source.is_relative_to(self.root):
                    raise RuntimeError('Source is outside the intended workspace: '+name)
                if not target.parent.is_dir():raise RuntimeError('Missing compatibility parent directory: '+str(target.parent))
                if target.exists() or target.is_symlink():raise RuntimeError('Refusing to overwrite an existing root file: '+str(target))
                aliases[name]=digest(source)
            self.record=dict(owner_pid=os.getpid(),child_pid=None,aliases=aliases)
            self._write()
            for name,source in self.sources.items():
                data=Path(source).read_bytes()
                if hashlib.sha256(data).hexdigest()!=aliases[name]:raise RuntimeError('Source changed during preparation: '+name)
                with (self.root/name).open('xb') as handle:handle.write(data)
            return self
        except BaseException:
            details=sys.exc_info()
            try:
                if self.record is not None:self._cleanup(self.record)
            finally:self.lock.__exit__(*details);self.lock=None
            raise

    def attach_child(self,pid):self.record['child_pid']=pid;self._write()

    def __exit__(self,kind,value,traceback):
        try:
            if not self.keep:self._cleanup(self.record)
        finally:
            if self.lock:self.lock.__exit__(kind,value,traceback)


def clean(root=ROOT):
    workspace=CompatibilityWorkspace(root,{})
    with command_lock(workspace.state/'command.lock'):workspace._recover()


def test_environment(root=ROOT):
    directory=Path(root).resolve()/'.cache/test_tmp';directory.mkdir(parents=True,exist_ok=True)
    return dict(os.environ,TMPDIR=str(directory),TEMP=str(directory),TMP=str(directory))


def run_python(arguments,root=ROOT,env=None,include_archived=False):
    root=Path(root).resolve()
    with CompatibilityWorkspace(root,runtime_sources(root,include_archived)) as workspace:
        child=subprocess.Popen([sys.executable,*arguments],cwd=root,env=env)
        workspace.attach_child(child.pid)
        try:return child.wait()
        except KeyboardInterrupt:
            # The terminal delivers Ctrl+C to the child as well. Allow it to
            # finish unwinding before removing paths used by its worker pool.
            try:child.wait(timeout=15)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                workspace.keep=True
                print('Research process is still stopping; compatibility files were preserved. Run `python -m hypershift clean` after it exits.',file=sys.stderr)
            return 130
