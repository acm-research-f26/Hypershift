"""Tiny JSON state helper for driver.sh (stdlib only).
  state.py FILE get KEY [DEFAULT]      dotted key; prints scalars raw, containers as JSON
  state.py FILE set KEY VALUE          VALUE parsed as JSON when possible, else string
  state.py FILE keys KEY               list child keys of an object, one per line
  state.py FILE init JSON              write JSON only if FILE does not exist
"""
import json, os, sys

f, cmd, *a = sys.argv[1:]
if cmd == "init":
    if not os.path.exists(f):
        open(f, "w").write(json.dumps(json.loads(a[0]), indent=1))
    sys.exit(0)
if cmd == "rmline":   # state.py QUEUEFILE rmline TEXT: delete the first line equal to TEXT (stripped); keeps comments
    lines = open(f).read().splitlines()
    for i, x in enumerate(lines):
        if x.strip() == a[0].strip():
            del lines[i]
            break
    open(f, "w").write("\n".join(lines) + ("\n" if lines else ""))
    sys.exit(0)
d = json.load(open(f)) if os.path.exists(f) else {}

def walk(key, create=False):
    cur = d
    parts = key.split(".") if key else []
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            if not create:
                return None, None
            cur[p] = {}
        cur = cur[p]
    return cur, (parts[-1] if parts else None)

if cmd == "get":
    cur, k = walk(a[0])
    v = cur.get(k) if cur is not None and k in cur else (a[1] if len(a) > 1 else "")
    print(json.dumps(v) if isinstance(v, (dict, list)) else ("true" if v is True else "false" if v is False else v))
elif cmd == "keys":
    cur, k = walk(a[0])
    v = cur.get(k) if cur is not None else None
    for x in (v or {}):
        print(x)
elif cmd == "set":
    cur, k = walk(a[0], create=True)
    try:
        v = json.loads(a[1])
    except Exception:
        v = a[1]
    cur[k] = v
    tmp = f + ".tmp"
    open(tmp, "w").write(json.dumps(d, indent=1))
    os.replace(tmp, f)
