# Workspace launcher

This standard-library package runs the organized historical scripts with their
original module names, root-relative paths and actual source bytes.

| Command | Purpose |
| --- | --- |
| `python -m hypershift list [filter]` | Find grouped sources |
| `python -m hypershift run <module> [arguments]` | Run an original script |
| `python -m hypershift test [unittest arguments]` | Run all or selected tests |
| `python -m hypershift python [Python arguments]` | Use historical imports interactively or with `-c`/`-m` |
| `python -m hypershift clean` | Recover unchanged temporary copies after an exited command |
| `python -m hypershift --archived list` | Include the older study archive explicitly |

Run these commands from the repository root. [workspace.py](workspace.py) creates
owned temporary root copies, starts the child interpreter, and removes unchanged
copies when it exits. It rejects existing root-file collisions and concurrent
launchers, and preserves edited copies or copies needed by a live child.
The canonical sources are in `latest/`, `research/` and `tests/`; edit and stage those
after commands finish.

The [full layout guide](../docs/PROJECT_LAYOUT.md) explains recovery and frozen-run
hash protection. [Launcher tests](../tests/test_workspace_layout.py) cover the
filesystem and subprocess behavior. This package contains no market models.

Latest protocols live in `latest/config/`. The launcher provides their original
`config/` paths temporarily, with the same collision, ownership and byte checks.
Archive mode also supplies archived source modules, old protocols and root
document aliases. It does not restore large historical datasets/runs or all old
publication paths; see [archive scope](../archive/studies_20261025/README.md).
