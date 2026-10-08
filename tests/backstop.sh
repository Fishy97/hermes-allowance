#!/bin/sh
# Backstop checks for bin/allowance-run (needs a systemd user session). Prints PASS/FAIL.
#  1. deadline: a command that would run 300s is killed, whole tree, at ALLOWANCE_SECONDS.
#  2. fork cap: no more than ALLOWANCE_TASKS processes can exist; anything already started
#     stays in the cgroup and is killed at the deadline even if its parent died early.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
fail=0

alive() { n=0; for p in $(cat "$1"); do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo "$n"; }

spawn() {  # spawn <count> <pidfile>: script that forks <count> sleepers and waits
  echo "for i in \$(seq $1); do sh -c 'echo \$\$ >> $2; exec sleep 300' 2>/dev/null & done; wait"
}

pids=$(mktemp); start=$(date +%s)
ALLOWANCE_SECONDS=3 ALLOWANCE_TASKS=64 "$here/bin/allowance-run" sh -c "$(spawn 5 "$pids")" 2>/dev/null
rc=$?; elapsed=$(( $(date +%s) - start )); sleep 1; left=$(alive "$pids")
if [ "$rc" -eq 143 ] && [ "$elapsed" -le 6 ] && [ "$left" -eq 0 ]; then r=PASS; else r=FAIL; fail=1; fi
echo "$r  deadline: exit $rc after ${elapsed}s, $(wc -l < "$pids") sleepers started, $left survivors"
rm -f "$pids"

pids=$(mktemp); start=$(date +%s)
ALLOWANCE_SECONDS=3 ALLOWANCE_TASKS=8 "$here/bin/allowance-run" sh -c "$(spawn 12 "$pids")" 2>/dev/null
rc=$?; started=$(wc -l < "$pids")
sleep $(( start + 5 - $(date +%s) )); left=$(alive "$pids")
if [ "$started" -le 7 ] && [ "$left" -eq 0 ]; then r=PASS; else r=FAIL; fail=1; fi
echo "$r  fork cap: $started of 12 sleepers started (TasksMax=8 incl. the shell), exit $rc, $left survivors after deadline"
rm -f "$pids"
exit $fail
