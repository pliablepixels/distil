#!/usr/bin/env bash
# Distil release driver — push, tag, publish to PyPI, refresh the Homebrew sha256.
#
# Outward-facing and mostly irreversible: every network step asks before it runs,
# and the script fails fast if anything looks off. Run it from the repo root on a
# clean `main`:
#
#   ./scripts/release.sh
#
# Prereqs:
#   - uv (build) and twine (upload) on PATH        # brew install uv twine
#   - PyPI credentials: a ~/.pypirc, or TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-…
#   - push rights to origin
#
# Flags:
#   --dry-run     print every action, change nothing (no push, no tag, no upload)
#   --skip-tests  skip the pytest gate (NOT recommended)
set -euo pipefail

DRY_RUN=0
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

# ---- pretty + safety helpers -------------------------------------------------
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

run() {  # echo + execute, or just echo under --dry-run
  printf '\033[90m$ %s\033[0m\n' "$*"
  [ "$DRY_RUN" -eq 1 ] && return 0
  eval "$*"
}

confirm() {  # confirm "message" ; aborts the step on anything but y/Y
  [ "$DRY_RUN" -eq 1 ] && { info "[dry-run] would ask: $1"; return 1; }
  printf '\033[33m? %s [y/N] \033[0m' "$1"
  read -r reply
  [[ "$reply" =~ ^[yY]$ ]]
}

# ---- derive version from the source of truth ---------------------------------
VERSION="$(grep -E '^version *= *"' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VERSION" ] || die "could not read version from pyproject.toml"
TAG="v$VERSION"
TARBALL_URL="https://github.com/dshakes/distil/archive/refs/tags/${TAG}.tar.gz"

bold "Distil release — ${TAG}$([ "$DRY_RUN" -eq 1 ] && echo '  (dry-run)')"
echo

# ---- preflight ---------------------------------------------------------------
bold "1/6  Preflight"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "not on main (on '$BRANCH')"
ok "on main"

[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash first"
ok "working tree clean"

# version agreement across every surface that ships a version string
INIT_V="$(grep -E '__version__' distil/__init__.py | sed -E 's/.*"([^"]+)".*/\1/')"
CITE_V="$(grep -E '^version:' CITATION.cff | sed -E 's/.*: *([0-9][^ ]*).*/\1/')"
[ "$INIT_V" = "$VERSION" ] || die "distil/__init__.py is $INIT_V, pyproject is $VERSION"
[ "$CITE_V" = "$VERSION" ] || die "CITATION.cff is $CITE_V, pyproject is $VERSION"
ok "version $VERSION consistent (pyproject, __init__, CITATION)"

git rev-parse "$TAG" >/dev/null 2>&1 && die "tag $TAG already exists — bump the version or delete the tag"
ok "tag $TAG is free"

grep -q "## \[$VERSION\]" CHANGELOG.md || die "CHANGELOG.md has no '## [$VERSION]' entry"
ok "CHANGELOG has a [$VERSION] entry"

if [ "$SKIP_TESTS" -eq 1 ]; then
  info "skipping tests (--skip-tests)"
else
  info "running test suite…"
  run ".venv/bin/python -m pytest -q" || die "tests failed — not releasing"
  ok "tests pass"
fi

# clean-install smoke test: build the wheel, install it into a throwaway venv with
# no index, and confirm the entrypoint + bundled corpus actually work.
info "clean-install smoke test…"
SMOKE="$(mktemp -d)"
trap 'rm -rf "$SMOKE"' EXIT
run "uv build --wheel -o '$SMOKE/wheel'"
if [ "$DRY_RUN" -eq 0 ]; then
  WHEEL="$(ls "$SMOKE"/wheel/distil_llm-"$VERSION"-py3-none-any.whl)"
  python3 -m venv "$SMOKE/venv"
  "$SMOKE/venv/bin/pip" install -q --no-index "$WHEEL"
  GOT="$("$SMOKE/venv/bin/distil" --version | awk '{print $2}')"
  [ "$GOT" = "$VERSION" ] || die "clean install reports $GOT, expected $VERSION"
  "$SMOKE/venv/bin/distil" bench >/dev/null || die "distil bench failed on a clean install"
  ok "clean install works: distil $VERSION, bench gate runs"
fi
echo

# ---- push main ---------------------------------------------------------------
bold "2/6  Push main"
if confirm "push 'main' to origin?"; then
  run "git push origin main"
  ok "main pushed"
else
  info "skipped push"
fi
echo

# ---- tag ---------------------------------------------------------------------
bold "3/6  Tag $TAG"
if confirm "create and push annotated tag $TAG?"; then
  run "git tag -a '$TAG' -m 'Distil $TAG'"
  run "git push origin '$TAG'"
  ok "$TAG pushed (this triggers the GitHub release tarball + pages deploy)"
else
  info "skipped tag — PyPI upload and Homebrew sha256 below need the tag; you can re-run later"
fi
echo

# ---- build + publish to PyPI -------------------------------------------------
bold "4/6  Build + publish to PyPI (distil-llm)"
if confirm "build sdist+wheel and upload $VERSION to PyPI? (irreversible — a version can't be re-uploaded)"; then
  command -v twine >/dev/null 2>&1 || die "twine not found (brew install twine), or upload manually with: uv publish"
  run "rm -rf dist/distil_llm-$VERSION*"
  run "uv build -o dist"
  run "twine check dist/distil_llm-$VERSION*"
  run "twine upload dist/distil_llm-$VERSION*"
  ok "uploaded distil-llm $VERSION to PyPI"
else
  info "skipped PyPI upload"
fi
echo

# ---- refresh Homebrew sha256 -------------------------------------------------
bold "5/6  Refresh Homebrew formula sha256"
FORMULA="packaging/homebrew/distil.rb"
if confirm "fetch $TAG tarball and patch the sha256 in $FORMULA?"; then
  if [ "$DRY_RUN" -eq 1 ]; then
    info "[dry-run] would curl $TARBALL_URL and shasum it"
  else
    info "fetching $TARBALL_URL …"
    SHA="$(curl -fsSL "$TARBALL_URL" | shasum -a 256 | awk '{print $1}')" \
      || die "could not fetch the tag tarball — is $TAG pushed yet?"
    [ "${#SHA}" -eq 64 ] || die "unexpected sha256: $SHA"
    # update sha + url + version in the formula
    sed -i.bak -E \
      -e "s|sha256 \"[^\"]*\"|sha256 \"$SHA\"|" \
      -e "s|/tags/v[0-9][^\"]*\.tar\.gz|/tags/${TAG}.tar.gz|" \
      -e "s|version \"[0-9][^\"]*\"|version \"$VERSION\"|" \
      "$FORMULA"
    rm -f "$FORMULA.bak"
    ok "patched $FORMULA → sha256 $SHA"
    if confirm "commit + push the formula bump?"; then
      run "git add '$FORMULA'"
      run "git commit -m 'release($VERSION): pin Homebrew sha256 to $TAG tarball'"
      run "git push origin main"
      ok "formula bump pushed"
    fi
  fi
else
  info "skipped Homebrew refresh"
fi
echo

# ---- done --------------------------------------------------------------------
bold "6/6  Done — manual follow-ups"
cat <<EOF
  • Homebrew TAP: this repo's $FORMULA is the source copy. Copy it into the actual
    tap repo so 'brew install dshakes/tap/distil' serves $VERSION:
        dshakes/homebrew-tap → Formula/distil.rb
  • GitHub release notes: paste the [$VERSION] section of CHANGELOG.md into
        https://github.com/dshakes/distil/releases/new?tag=$TAG
  • Verify the live install:  pipx install distil-llm && distil --version   # → $VERSION
  • Docs site redeploys from main via the pages workflow — confirm it's green.
EOF
ok "release driver finished"
