#!/usr/bin/env bash
# Usage: refresh-bot-prs.sh PREFIX
#
# Main's ruleset requires a PR to be up to date before it merges, so a bot PR that falls behind
# main never merges, auto-merge or not. For every open PR whose head branch starts with PREFIX
# (`rates/sync-` or `catalog/`) and is behind main, when every commit on it since main is the
# bot's: rebase it onto main, force-push with lease, re-enable auto-merge if it had it, and
# dispatch CI on it (a GITHUB_TOKEN push starts no pull_request run). A branch carrying anyone
# else's commit is left alone.
#
# A rebase that conflicts is regenerated instead: a `rates/sync-*` branch by regenerating the rate
# table on main from the branch's snapshots; a `catalog/*` branch is left for its workflow (which
# rebuilds every current finding's branch from main on each run), with a comment.
#
# Needs GH_TOKEN, GITHUB_REPOSITORY, and a full clone (actions/checkout `fetch-depth: 0`).
set -euo pipefail
prefix=$1
bot="41898282+github-actions[bot]@users.noreply.github.com"
remote="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git config user.name "github-actions[bot]"
git config user.email "$bot"
git fetch -q "$remote" "+refs/heads/main:refs/remotes/origin/main"

regenerate_rates() {
  local branch=$1
  git checkout -q -B "refresh/$branch" origin/main
  git rm -rq --cached verify/rates_sources
  rm -rf verify/rates_sources
  git checkout -q "origin/$branch" -- verify/rates_sources
  cargo run -q -p beyond-ai-rates-sync -- generate </dev/null
  cargo run -q -p beyond-ai-rates-sync -- rate-version </dev/null
  git add -A -- verify/rates_sources crates/providers/src/rates/generated.rs \
    crates/providers/src/rates.rs verify/pricing_vectors.json
  git commit -q -m "$(git log -1 --format=%s "origin/$branch")" \
    -m "Regenerated on main from this branch's snapshots by refresh-bot-prs.sh."
}

prs=$(gh pr list --state open --limit 200 --json number,headRefName,autoMergeRequest \
  --jq "[.[] | select(.headRefName | startswith(\"$prefix\"))] | sort_by(.number) | .[] | \"\(.number) \(.headRefName) \(.autoMergeRequest != null)\"")
while read -r pr branch auto; do
  [ -n "${pr:-}" ] || continue
  git fetch -q "$remote" "+refs/heads/$branch:refs/remotes/origin/$branch"
  behind=$(git rev-list --count "origin/$branch..origin/main")
  if [ "$behind" = 0 ]; then
    echo "#$pr ($branch): up to date with main"
    continue
  fi
  if git log --format='%ae%n%ce' "origin/main..origin/$branch" | grep -qvxF "$bot"; then
    echo "#$pr ($branch): $behind behind main, but it has a commit from a human: left alone"
    continue
  fi
  lease=$(git rev-parse "origin/$branch")
  git checkout -q -B "refresh/$branch" "origin/$branch"
  if ! git rebase -q origin/main; then
    git rebase --abort
    case "$prefix" in
      rates/*) regenerate_rates "$branch" ;;
      *)
        gh pr comment "$pr" --body "This branch is $behind commits behind main and no longer rebases cleanly. The next catalog-drift run rebuilds it from main if its finding still holds; otherwise close it."
        continue
        ;;
    esac
  fi
  git push -q --force-with-lease="refs/heads/$branch:$lease" "$remote" "HEAD:refs/heads/$branch"
  gh workflow run ci.yml --ref "$branch"
  if [ "$auto" = true ]; then
    gh pr merge "$pr" --auto --squash --delete-branch
  fi
  echo "#$pr ($branch): was $behind behind main; refreshed and CI dispatched"
done <<<"$prs"
git checkout -q --detach origin/main
