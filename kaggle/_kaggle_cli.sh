#!/usr/bin/env bash
# Sourced by launch.sh / fetch.sh. Defines: KG (kaggle CLI command), KUSER (username), and checks the credentials.
# The CLI is run as `python -m kaggle.cli` from a private venv (kaggle/.venv-kaggle) so the project .venv is never touched.
# (The kaggle.exe launcher can be blocked by Smart App Control; the module form is not.)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KPY="$ROOT/kaggle/.venv-kaggle/Scripts/python.exe"
if [ ! -x "$KPY" ]; then
  python3 -m venv "$ROOT/kaggle/.venv-kaggle" || python -m venv "$ROOT/kaggle/.venv-kaggle" || exit 1
  "$KPY" -m pip install -q kaggle || exit 1
fi
kg() { "$KPY" -m kaggle.cli "$@"; }
KDIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
if [ ! -f "$KDIR/kaggle.json" ] && [ ! -f "$KDIR/access_token" ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "NO KAGGLE CREDENTIALS. Put kaggle.json (or access_token) in $KDIR (Windows: C:\\Users\\Hi\\.kaggle\\) or set KAGGLE_API_TOKEN." >&2
  echo "Token: https://www.kaggle.com/settings/api -> Create New Token. Everything else is already prepared." >&2
  exit 2
fi
KUSER="${KAGGLE_USER:-}"
if [ -z "$KUSER" ] && [ -f "$KDIR/kaggle.json" ]; then
  KUSER=$("$KPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['username'])" "$KDIR/kaggle.json")
fi
if [ -z "$KUSER" ]; then echo "Set KAGGLE_USER=<your kaggle username> (needed with access_token auth)." >&2; exit 2; fi
