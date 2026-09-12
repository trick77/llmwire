#!/usr/bin/env bash
# hack/secret-scan.sh [--staged]
#
# Refuses API keys and .env files entering the repository.
#
# This module's eval suite talks to real, paid endpoints with real keys, and its
# findings notes are committed artefacts — so a key has three ways in that a
# .gitignore alone does not close: pasted into a findings note, pasted into a
# test as a "temporary" literal, or captured verbatim inside a quoted upstream
# error body. This catches all three.
#
# --staged scans the staged diff (the pre-commit hook). Without it, the whole
# tracked tree is scanned, which is what CI runs: a hook only protects the
# machine it is installed on, and hooks are not shared by git.
set -euo pipefail
export LC_ALL=C

# Key shapes actually in use by the endpoints this module talks to:
#   sk-…  OpenAI, Z.ai, MiMo pay-as-you-go, LiteLLM virtual keys
#   tp-…  MiMo Token Plan
# The length bound is what keeps prose like "sk-" or a bare "tp-" from tripping
# it; a real key is far longer than 16 characters.
PATTERN='(sk|tp)-[A-Za-z0-9_-]{16,}'

die() { printf 'secret-scan: %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--staged" ]; then
    files=$(git diff --cached --name-only --diff-filter=ACM)
    # A staged .env is a mistake .gitignore cannot catch, because `git add -f`
    # and an already-tracked file both bypass it.
    if printf '%s\n' "$files" | grep -qx '.env'; then
        die "refusing to commit .env — keys belong in the environment, never in the tree"
    fi
    [ -n "$files" ] || exit 0
    if hits=$(git diff --cached -U0 -- $files | grep -nE "^\+.*$PATTERN" || true); [ -n "$hits" ]; then
        printf '%s\n' "$hits" >&2
        die "staged change contains something shaped like an API key"
    fi
else
    if git ls-files | grep -qx '.env'; then
        die ".env is tracked — it must never be"
    fi
    if hits=$(git grep -nE "$PATTERN" -- . ':!hack/secret-scan.sh' || true); [ -n "$hits" ]; then
        printf '%s\n' "$hits" >&2
        die "tracked file contains something shaped like an API key"
    fi
fi
