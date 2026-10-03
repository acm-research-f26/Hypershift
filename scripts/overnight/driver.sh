#!/usr/bin/env bash
# Zero-Claude-token overnight driver. One idempotent tick per run (Task Scheduler runs it every 30 min; see install.sh).
#   bash scripts/overnight/driver.sh [--dry-run] [--reset]
# Per tick: (1) fetch+merge finished Kaggle kernels, (2) analysis kernel, (3) launch next kaggle/queue.txt preset,
# (4) Codex edits the verdict docs, (5) git push, (6) ALL DONE check. State: scripts/overnight/state.json.
# Log: results/logs/overnight.log. CPU only; no Claude involved.
export PATH="/usr/bin:/mingw64/bin:/c/Users/Hi/AppData/Roaming/npm:$PATH"
export CUDA_VISIBLE_DEVICES=-1 KAGGLE_USER="${KAGGLE_USER:-tomphamdustry}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
D=scripts/overnight; STATE=$D/state.json; LOG=results/logs/overnight.log; LOCK=$D/.lock
QUEUE="${QUEUE_FILE:-kaggle/queue.txt}"; BRANCH=tom-shlom
PY="$ROOT/kaggle/.venv-kaggle/Scripts/python.exe"      # light stdlib-only interpreter for the helpers
KG=("$PY" -m kaggle.cli)
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"                  # ~/.codex/config.toml's gpt-6-sol is rejected for ChatGPT accounts (codex-cli 0.124.0)
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
mkdir -p results/logs "$D"
log() { printf '%s [%s] %s\n' "$(date '+%F %T')" "$([ $DRY = 1 ] && echo DRY || echo tick)" "$*" | tee -a "$LOG"; }
spy() { local i out; for i in 1 2 3; do out=$("$PY" "$@" 2>/dev/null | tr -d "\r"; exit ${PIPESTATUS[0]}) && { echo "$out"; return 0; }; sleep 3; done; return 1; }   # retry: Smart App Control blocks sporadically
sget() { local v; v=$(spy $D/state.py $STATE get "$@") || log "WARN: state helper failed (get $1)"; echo "$v"; }
sset() { [ $DRY = 1 ] && return 0; spy $D/state.py $STATE set "$@" > /dev/null; }
skeys() { spy $D/state.py $STATE keys "$1"; }

if [ "${1:-}" = "--reset" ]; then sset done false; log "done flag reset"; exit 0; fi

# --- lock (a tick killed by the 25 min task limit leaves a lock; treat it as stale after 28 min) ---
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +28 2>/dev/null)" ]; then rmdir "$LOCK" 2>/dev/null && log "removed stale lock"; fi
mkdir "$LOCK" 2>/dev/null || { log "another tick is running (lock held); exiting"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

"$PY" $D/state.py $STATE init '{"done": false, "kernels": {
 "r5f": {"slug": "hypershift-run-r5f", "origin": "manual", "fetched": false, "attempts": 0},
 "r8f": {"slug": "hypershift-run-r8f", "origin": "manual", "fetched": false, "attempts": 0}},
 "analysis": {"tag": "r5f-an", "stem": "F_r5f", "launched": false, "fetched": false, "attempts": 0},
 "docs_pending": "", "docs_attempts": 0, "last_pushed_head": "", "launch_fail_sig": "", "launch_fail_sig_line": "", "pending_launch": ""}'

if [ "$(sget done false)" = true ]; then log "done=true; no-op (use --reset to re-arm)"; exit 0; fi
log "tick start (HEAD $(git rev-parse --short HEAD))"

kstatus() {   # $1 slug -> COMPLETE|ERROR|RUNNING|QUEUED|CANCEL...|UNKNOWN (Smart App Control may block the CLI: retry)
  local out st i
  for i in 1 2 3; do
    out=$("${KG[@]}" kernels status "tomphamdustry/$1" 2>&1 | tr -d '\r')
    st=$(echo "$out" | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1 | sed 's/.*\.//')
    [ -n "$st" ] && { echo "$st"; return; }
    sleep 8
  done
  echo UNKNOWN
}
terminal() { case "$1" in COMPLETE|ERROR|CANCEL*) return 0;; esac; return 1; }
live() { case "$1" in RUNNING|QUEUED|NEW) return 0;; esac; return 1; }
retry() { local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; sleep 10; done; return 1; }
add_pending() { local cur; cur=$(sget docs_pending ""); case " $cur " in *" $1 "*) ;; *) sset docs_pending "$cur $1";; esac; }
commit_paths() {   # $1 message, rest = paths. Pathspec commit: never sweeps up other staged/unstaged files.
  local msg=$1 f p=(); shift
  for f in "$@"; do [ -e "$f" ] && p+=("$f"); done
  [ ${#p[@]} -eq 0 ] && return 1
  [ -z "$(git status --porcelain -- "${p[@]}")" ] && return 1
  if [ $DRY = 1 ]; then log "would commit: ${p[*]}"; return 0; fi
  git add -- "${p[@]}" && git commit -q -m "$msg" -- "${p[@]}" && log "committed $(git rev-parse --short HEAD): $msg"
}
track_launched() {   # $1 tag $2 raw queue line
  sset kernels.$1 "{\"slug\": \"hypershift-run-${1//_/-}\", \"origin\": \"driver\", \"fetched\": false, \"attempts\": 0, \"state\": \"RUNNING\", \"line\": \"$2\"}"
  "$PY" $D/state.py "$QUEUE" rmline "$2"
}

# ============ 1. collect finished kernels ============
for tag in $(skeys kernels); do
  [ "$(sget kernels.$tag.fetched false)" = true ] && continue
  slug=$(sget kernels.$tag.slug); st=$(kstatus "$slug"); sset kernels.$tag.state "$st"; eval "KS_${tag//-/_}=$st"
  log "kernel $tag ($slug): $st"
  terminal "$st" || continue
  if [ $DRY = 1 ]; then log "would: fetch.sh $tag; merge zip (merge_results.sh, fallback merge_zip.py); copy analysis md; mark fetched"; continue; fi
  out=kaggle/build/out_$tag; zip=$out/results_$tag.zip
  bash kaggle/fetch.sh "$tag" >> "$LOG" 2>&1 || retry 2 bash kaggle/fetch.sh "$tag" >> "$LOG" 2>&1
  if [ -f "$zip" ]; then
    merged=0
    if bash kaggle/merge_results.sh "$zip" >> "$LOG" 2>&1; then merged=1
    else
      log "merge_results.sh failed (PermissionError?); using scripts/overnight/merge_zip.py (no os.rename)"
      "$PY" $D/merge_zip.py "$zip" >> "$LOG" 2>&1 && merged=1
    fi
    if [ $merged = 1 ]; then
      "$PY" $D/merge_zip.py "$zip" --no-merge 2>&1 | tee -a "$LOG" | sed -n 's/^DOC //p' >> "$D/.newdocs"
      sset kernels.$tag.fetched true; log "kernel $tag collected from $zip"
    else
      n=$(( $(sget kernels.$tag.attempts 0) + 1 )); sset kernels.$tag.attempts $n; log "kernel $tag: merge failed (attempt $n)"
      if [ $n -ge 3 ]; then sset kernels.$tag.fetched true; log "kernel $tag: giving up on merge after 3 attempts; MANUAL ACTION NEEDED: $zip"; fi
    fi
  elif [ "$st" = COMPLETE ]; then
    n=$(( $(sget kernels.$tag.attempts 0) + 1 )); sset kernels.$tag.attempts $n; log "kernel $tag COMPLETE but no zip downloaded (attempt $n)"
    if [ $n -ge 3 ]; then sset kernels.$tag.fetched true; log "kernel $tag: giving up; MANUAL ACTION NEEDED"; fi
  else sset kernels.$tag.fetched true; log "kernel $tag ended $st with no results zip; marked collected"; fi
done
if [ -s "$D/.newdocs" ]; then
  for md in $(sort -u "$D/.newdocs"); do add_pending "$md"; done
  rm -f "$D/.newdocs"
fi

# ============ 2. analysis kernel ============
AT=$(sget analysis.tag r5f-an); STEM=$(sget analysis.stem F_r5f)
s5=${KS_r5f:-$(sget kernels.r5f.state UNKNOWN)}; s8=${KS_r8f:-$(sget kernels.r8f.state UNKNOWN)}
if [ "$(sget analysis.launched false)" != true ]; then
  if terminal "$s5" && terminal "$s8"; then
    if [ $DRY = 1 ]; then log "would: bash kaggle/launch_analysis.sh (r5f=$s5 r8f=$s8)"
    else
      log "r5f=$s5 r8f=$s8 both terminal: launching analysis kernel"
      if timeout 1300 bash kaggle/launch_analysis.sh >> "$LOG" 2>&1; then sset analysis.launched true; log "analysis kernel launched"
      else
        ast=$(kstatus "hypershift-run-$AT")
        if live "$ast"; then sset analysis.launched true; log "launch_analysis.sh returned nonzero but kernel is $ast; treating as launched"
        else log "launch_analysis.sh failed (kernel status $ast); will retry next tick"; fi
      fi
    fi
  else log "analysis waiting: r5f=$s5 r8f=$s8"; fi
elif [ "$(sget analysis.fetched false)" != true ]; then
  ast=$(kstatus "hypershift-run-$AT"); log "analysis kernel $AT: $ast"
  if terminal "$ast"; then
    if [ $DRY = 1 ]; then log "would: bash kaggle/fetch_analysis.sh $AT $STEM --install"
    elif bash kaggle/fetch_analysis.sh "$AT" "$STEM" --install >> "$LOG" 2>&1; then
      sset analysis.fetched true
      [ -f docs/phase1_5/${STEM}_results.md ] && add_pending docs/phase1_5/${STEM}_results.md
      log "analysis fetched ($ast)"
    else
      n=$(( $(sget analysis.attempts 0) + 1 )); sset analysis.attempts $n; log "fetch_analysis failed (attempt $n)"
      [ $n -ge 3 ] && sset analysis.fetched true
    fi
  fi
fi

# ============ 3. queue (at most one preset per tick) ============
pend=$(sget pending_launch "")
if [ -n "$pend" ]; then   # a previous tick died mid-launch (task time limit): did the kernel get pushed?
  ptag=${pend%%|*}; pline=${pend#*|}; pst=$(kstatus "hypershift-run-${ptag//_/-}")
  if live "$pst"; then
    log "recovered interrupted launch $ptag ($pst)"
    if [ $DRY = 0 ]; then track_launched "$ptag" "$pline"; sset pending_launch ""; fi
  else log "interrupted launch $ptag not found on Kaggle ($pst); will retry the line"; sset pending_launch ""; fi
fi
QEMPTY=0
if [ ! -f "$QUEUE" ]; then log "queue: $QUEUE missing (waiting for it to be written)"; QEMPTY=-1
else
  mapfile -t QL < <(tr -d '\r' < "$QUEUE" | grep -v '^[[:space:]]*#' | sed 's/[[:space:]]*#.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^$')
  mapfile -t QRAW < <(tr -d '\r' < "$QUEUE" | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$')
  log "queue: ${#QL[@]} line(s): ${QL[*]:-<empty>}"
  running_sig=$(for t in $(skeys kernels); do if [ "$(sget kernels.$t.fetched false)" != true ] && ! terminal "$(sget kernels.$t.state UNKNOWN)"; then echo "$t"; fi; done | sort | tr '\n' ',')
  for i in "${!QL[@]}"; do
    line=${QL[$i]}; raw=${QRAW[$i]}; set -- $line; preset=$1; tag=${2:-$1}
    if ! grep -q "\"$preset\":" kaggle/run_kaggle.py; then log "queue: '$preset' is not a preset in kaggle/run_kaggle.py COMMANDS; skipping this line (left in queue)"; continue; fi
    if [ "$preset" = r8f-top ] && ! terminal "$s8"; then log "queue: r8f-top waits for r8f (state $s8)"; continue; fi
    if [ "$(sget launch_fail_sig_line "")" = "$line" ] && [ "$(sget launch_fail_sig "")" = "$running_sig" ]; then
      log "queue: last launch of '$line' was refused and no tracked kernel changed state since; waiting"; break; fi
    if [ $DRY = 1 ]; then log "would: bash kaggle/launch.sh $preset $tag ; then remove line '$raw' and track kernel hypershift-run-${tag//_/-}"; break; fi
    log "queue: launching 'bash kaggle/launch.sh $preset $tag'"
    sset pending_launch "$tag|$raw"
    ok=0
    if timeout 1300 bash kaggle/launch.sh "$preset" "$tag" >> "$LOG" 2>&1; then ok=1
    else lst=$(kstatus "hypershift-run-${tag//_/-}"); if live "$lst"; then ok=1; log "launch.sh nonzero but kernel is $lst; treating as launched"; fi; fi
    if [ $ok = 1 ]; then
      track_launched "$tag" "$raw"; sset pending_launch ""; sset launch_fail_sig ""; sset launch_fail_sig_line ""
      log "queue: launched $tag, line removed"
    else
      sset pending_launch ""; sset launch_fail_sig "$running_sig"; sset launch_fail_sig_line "$line"
      log "queue: launch of '$line' failed/refused; line kept, retry when a tracked kernel changes state"
    fi
    break   # one attempt per tick
  done
  [ ${#QL[@]} -eq 0 ] && QEMPTY=1
fi

# ============ 4. docs via Codex ============
pending=$(sget docs_pending "" | xargs)
if [ -n "$pending" ]; then
  # new result docs/figures are committed first (so the push carries them even if Codex fails)
  commit_paths "docs(phase1.5): overnight results ($pending)" $pending docs/figures/${STEM}_fig.png
  if [ $DRY = 1 ]; then log "would: codex exec -m $CODEX_MODEL -s workspace-write on: $pending"
  elif [ "$(sget docs_attempts 0)" -ge 3 ]; then
    log "codex: 3 attempts used for '$pending'; giving up (tracker/summary left for manual update)"; sset docs_pending ""; sset docs_attempts 0
  else
    sset docs_attempts $(( $(sget docs_attempts 0) + 1 ))
    PROMPT="You are updating project docs in the Hypershift repo (cwd). Read CLAUDE.md first (and AGENTS.md / Codex.md if they exist). Rules: the paper source of truth is docs/paper/icdm22-think.pdf (pp. 849-854); cite page and section/equation/table for any paper detail; anything the PDF does not state must be labelled 'INFERRED (not in paper)' or 'UNKNOWN'; never guess. New result document(s) arrived: $pending. Read them fully. Then edit ONLY these two files: docs/PHASE1_TRACKER.md and docs/phase1_5/PHASE1_5_SUMMARY.md (fill the PENDING section 5 'Final verdict' and update the tracker verdict line and entries). Use the repo's exact verdict words only: STRONG, SEED-ROBUST ONLY, NO EVIDENCE, INSUFFICIENT SEEDS, taken from the results document; report numbers exactly as written there, for norm=paper and norm=train and for validation-selected epoch vs test_oracle_sr, and note which seeds finished. If a result document lacks what you need, write UNKNOWN or PENDING rather than inventing. No em dashes, no emojis. Do NOT run git commit or git add (the sandbox blocks .git writes; the driver commits), do NOT push, do NOT run training, do NOT use Kaggle or the network, do NOT edit any other file. End with a 3-line summary of what you changed."
    cdx=results/logs/codex_$(date +%Y%m%d_%H%M%S).txt
    log "codex: running (log $cdx)"
    timeout 1200 codex exec -m "$CODEX_MODEL" -c model_reasoning_effort=medium -s workspace-write --ephemeral -C "$ROOT" -o "$cdx.last" "$PROMPT" < /dev/null > "$cdx" 2>&1; crc=$?
    log "codex: exit $crc; last message: $(tr '\n' ' ' < "$cdx.last" 2>/dev/null | cut -c1-300)"
    others=$(git status --porcelain -- docs | grep -v -E 'PHASE1_TRACKER.md|PHASE1_5_SUMMARY.md|F_r5f|docs/figures|AGENTS.md|Codex.md' | head -5)
    [ -n "$others" ] && log "WARNING: Codex touched other docs (left uncommitted): $others"
    if [ $crc = 0 ]; then
      commit_paths "docs: record overnight verdict in tracker and Phase 1.5 summary (written by Codex via overnight driver)" docs/PHASE1_TRACKER.md docs/phase1_5/PHASE1_5_SUMMARY.md || log "codex: no change to tracker/summary to commit"
      sset docs_pending ""; sset docs_attempts 0
    else log "codex failed; will retry next tick (attempt $(sget docs_attempts 0)/3)"; fi
  fi
fi

# ============ 5. push ============
head=$(git rev-parse HEAD); remote=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
if [ "$head" != "$(sget last_pushed_head "")" ] && [ "$head" != "$remote" ]; then
  if [ $DRY = 1 ]; then log "would: git push origin $BRANCH (HEAD ${head:0:7} ahead of origin ${remote:0:7})"
  elif timeout 300 git push origin "$BRANCH" >> "$LOG" 2>&1; then sset last_pushed_head "$head"; log "pushed ${head:0:7} to origin/$BRANCH"
  else log "git push FAILED (will retry next tick)"; fi
else log "push: nothing new (HEAD ${head:0:7})"; fi

# ============ 6. done? ============
alldone=1
for t in $(skeys kernels); do [ "$(sget kernels.$t.fetched false)" = true ] || alldone=0; done
[ "$(sget analysis.launched false)" = true ] && [ "$(sget analysis.fetched false)" = true ] || alldone=0
[ -z "$(sget docs_pending "" | xargs)" ] || alldone=0
[ -z "$(sget pending_launch "")" ] || alldone=0
if [ "$QEMPTY" = 1 ] && [ $alldone = 1 ]; then
  if [ $DRY = 1 ]; then log "would: ALL DONE"; else sset done true; log "ALL DONE"; fi
fi
log "tick end"
