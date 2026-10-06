# Archived experiments

Superseded work was moved here on October 5, 2026 to keep the active project
focused on the controlled THINK/literature architecture research.

`cleanup_manifest.json` records each original and destination path. `legacy/`
preserves the original GNN/direction tutorials, hourly and hyperbolicity audits,
the earlier THINK reconstruction runner, tests, data and result folders. The old
README and THINK backbone were also copied here for historical context.

Nothing in this archive is required by the current training/evaluation pipeline.
Frozen universe lists needed by current downloaders live in `config/`.
Generated archived data and runs remain Git-ignored, like their original paths.

From the project root, archived unit tests can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s archive\legacy -p "test_*.py"
```

Old scripts generally expect to run from `archive/legacy/` because their relative
`data/` and `runs/` paths were preserved there. Their original documentation is
historical and is not the current project entry point.
