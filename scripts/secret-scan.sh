#!/usr/bin/env bash
# scripts/secret-scan.sh [--staged]
#
# Refuses API keys and .env files entering the repository.
#
# This module's eval suite talks to real, paid endpoints with real keys, and its
# findings notes are committed artefacts — so a key has three ways in that a
# .gitignore alone does not close: pasted into a findings note, pasted into a
# test as a "temporary" literal, or captured verbatim inside a quoted upstream
# error body. This catches all three.
#
# Without arguments (a ship.yaml step, so ship check runs it on every push and
# in CI): the whole tracked tree, plus every line the commits not yet on
# origin/master add — a key added in one commit and removed in the next is
# clean at the tip but still reaches origin in history.
#
# --staged scans the staged diff only, for a pre-commit hook if one is wired
# up by hand (none is installed by default).
set -euo pipefail
export LC_ALL=C

# Key shapes actually in use by the endpoints this module talks to:
#   sk-…  OpenAI, Z.ai, MiMo pay-as-you-go, LiteLLM virtual keys
#   tp-…  MiMo Token Plan
# The length bound is what keeps prose like "sk-" or a bare "tp-" from tripping
# it; a real key is far longer than 16 characters.
PATTERN='(sk|tp)-[A-Za-z0-9_-]{16,}'
# .env at any depth, as .gitignore's `.env` ignores it.
ENV='(^|/)\.env$'

die() { printf 'secret-scan: %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--staged" ]; then
    files=$(git diff --cached --name-only --diff-filter=ACM)
    # A staged .env is a mistake .gitignore cannot catch, because `git add -f`
    # and an already-tracked file both bypass it.
    if printf '%s\n' "$files" | grep -E "$ENV" >/dev/null; then
        die "refusing to commit .env — keys belong in the environment, never in the tree"
    fi
    [ -n "$files" ] || exit 0
    if hits=$(git diff --no-color --cached -U0 -- $files | grep -nE "^\+.*$PATTERN" || true); [ -n "$hits" ]; then
        printf '%s\n' "$hits" >&2
        die "staged change contains something shaped like an API key"
    fi
else
    if git ls-files | grep -E "$ENV" >/dev/null; then
        die ".env is tracked — it must never be"
    fi
    if hits=$(git grep -nE "$PATTERN" -- . ':!scripts/secret-scan.sh' || true); [ -n "$hits" ]; then
        printf '%s\n' "$hits" >&2
        die "tracked file contains something shaped like an API key"
    fi
    # Unpushed history. No origin/master (a fresh clone mid-setup): the tree scan
    # above is all there is. Each git log is a plain assignment, so a failing
    # one ends the script (set -e) instead of reading as "nothing found".
    # --no-color: color.ui=always would wrap every + line in escapes.
    # --diff-merges=first-parent: a merge's own changes count. --no-renames: a
    # .env renamed into place shows as added.
    if base=$(git merge-base HEAD origin/master 2>/dev/null); then
        walk=(--no-color --no-renames --diff-merges=first-parent "$base"..HEAD)
        added=$(git log --format= --name-only --diff-filter=A "${walk[@]}")
        if printf '%s\n' "$added" | grep -E "$ENV" >/dev/null; then
            die "a commit not yet on origin/master adds .env — rewrite it out before pushing"
        fi
        # A pasted upstream error body can land in a commit message too.
        messages=$(git log --format='commit %h%n%B' "${walk[@]}")
        patches=$(git log -p --format='commit %h' "${walk[@]}" -- . ':!scripts/secret-scan.sh')
        if hits=$({ printf '%s\n' "$messages" | grep -E "^commit |$PATTERN"
            printf '%s\n' "$patches" | grep -E "^commit |^\+.*$PATTERN"; } |
            grep -B1 -vE '^commit ' | grep -v '^--$' || true); [ -n "$hits" ]; then
            printf '%s\n' "$hits" >&2
            die "a commit not yet on origin/master carries something shaped like an API key — rewrite it out before pushing"
        fi
    fi
fi
