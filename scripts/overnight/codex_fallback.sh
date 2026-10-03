#!/usr/bin/env bash
# Phase 1.5a Codex fallback (Task Scheduler Hypershift_codex_fallback, every 30 min, zero Claude tokens).
# Codex runs the next plan task only when Claude cannot: Claude usage >= 95% (5h) / 98% (7d), or the Claude status line has
# not refreshed for 3 h (session closed). Never while someone is working: skipped if any Phase 1.5a file changed in the
# last 30 min. One task per tick. Codex's sandbox cannot write .git, so this script commits (pathspec only) and pushes.
# Stops for good once docs/phase1_5a/DONE exists. Log: results/logs/codex_fallback.log.
cd "$(dirname "$0")/../.." || exit 1
LOG=results/logs/codex_fallback.log; BRANCH=tom-shlom; MODEL="${CODEX_MODEL:-gpt-5.5}"
PLAN=docs/superpowers/plans/2026-10-03-phase1-5a-sharpe2-forensics.md; PROG=docs/phase1_5a/PROGRESS.md
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
[ -f docs/phase1_5a/DONE ] && exit 0
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { log "not on $BRANCH; skip"; exit 0; }

U="$HOME/.claude/usage-latest.json"
read -r FIVE SEVEN <<< "$(node -e 'try{const r=require(process.argv[1]).rate_limits||{};console.log(((r.five_hour||{}).used_percentage??0)+" "+((r.seven_day||{}).used_percentage??0))}catch(e){console.log("0 0")}' "$U" 2>/dev/null)"
AGE=$(( $(date +%s) - $(stat -c %Y "$U" 2>/dev/null || echo 0) ))
WHY=""
[ "${FIVE%.*}" -ge 95 ] 2>/dev/null && WHY="claude 5h usage ${FIVE}%"
[ "${SEVEN%.*}" -ge 98 ] 2>/dev/null && WHY="claude 7d usage ${SEVEN}%"
[ "$AGE" -gt 10800 ] && WHY="${WHY:-claude status line stale $((AGE/60)) min}"
[ -z "$WHY" ] && exit 0                                   # Claude is fine: do nothing, log nothing

PATHS="src/hypershift/eval/forensics.py scripts/forensic_2017.py tests/test_forensics.py tests/test_forensics_artifacts.py tests/test_think_equivariance.py docs/phase1_5a .gitignore"
if [ -n "$(find $PATHS docs/figures -newermt '-30 minutes' -type f 2>/dev/null | grep -E 'phase1_5a|forensic|equivariance|gitignore' | head -1)" ]; then
  log "trigger ($WHY) but Phase 1.5a files changed <30 min ago: someone is working; skip"; exit 0
fi
NB=$(grep -c "BLOCKED" "$PROG" 2>/dev/null); NB=${NB:-0}
[ "$NB" -ge 2 ] && { log "2+ BLOCKED entries in PROGRESS.md: MANUAL ACTION needed; skip"; exit 0; }

PROMPT="You are executing a written plan in the Hypershift repo (branch $BRANCH). Read CLAUDE.md, the plan $PLAN, and $PROG.
Execute exactly ONE task: the first task in the order T1, T2, T2B, T5, T3, T4, T6, T7, T8 that is not marked done in $PROG
(if Gate A in docs/phase1_5a/REPORT_2017.md says outcome 1, 2 or 3, trim batch 2 exactly as the plan's Gate A step says).
Follow the plan's code, grids, tests and constraints exactly. CPU only: prefix every python command with CUDA_VISIBLE_DEVICES=-1;
use .venv/Scripts/python.exe. Never write under results/R5_*, results/R8_*, results/POC_*; never launch Kaggle or GPU work.
You cannot use git (the wrapper commits); do not try. Uncommitted work from an interrupted earlier attempt may exist: inspect it and finish it.
When the task's tests pass and its outputs are written, append to $PROG: 'T<id> done <YYYY-MM-DD HH:MM> by codex: <one-line result>'.
If you cannot complete it, append 'T<id> BLOCKED: <reason>' and stop. If you completed T8, also create the empty file docs/phase1_5a/DONE.
Final message: task id, tests run with pass/fail counts, files changed."
cdx=results/logs/codex_1_5a_$(date +%Y%m%d_%H%M%S).txt
log "trigger ($WHY): codex exec (log $cdx)"
timeout 3300 codex exec -m "$MODEL" -c model_reasoning_effort=medium -s workspace-write --ephemeral -C "$PWD" -o "$cdx.last" "$PROMPT" < /dev/null > "$cdx" 2>&1; rc=$?
log "codex exit $rc: $(tr '\n' ' ' < "$cdx.last" 2>/dev/null | cut -c1-300)"

ex=(); for p in $PATHS docs/figures/phase1_5a_*.png docs/phase1_5/PHASE1_5_SUMMARY.md docs/PHASE1_TRACKER.md CLAUDE.md; do [ -e "$p" ] && ex+=("$p"); done
git add -- "${ex[@]}" 2>>"$LOG"
if ! git diff --cached --quiet -- "${ex[@]}"; then
  git commit -q -m "feat(phase1.5a): $(grep -E '^T[0-9A-Z]+ (done|BLOCKED)' "$PROG" | tail -1 | cut -c1-120) (Codex fallback)" -- "${ex[@]}" \
    && log "committed $(git rev-parse --short HEAD)"
  timeout 300 git push origin "$BRANCH" >> "$LOG" 2>&1 && log "pushed" || log "push FAILED"
else log "nothing to commit"; fi
