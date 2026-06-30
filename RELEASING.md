# Releasing Distil

Distil publishes to PyPI via **Trusted Publishing (OIDC)** — CI proves its identity to
PyPI directly, so **no API token exists anywhere** (not on a laptop, not in a GitHub
secret). You push a `v*` tag; `.github/workflows/release.yml` gates, builds, attaches
GitHub release artifacts, and publishes to PyPI.

## One-time setup (do once, never again)

1. **PyPI pending publisher** — pypi.org → project `distil-llm` → *Publishing* → *Add a
   pending publisher*:
   - Owner: `dshakes` · Repository: `distil` · Workflow: `release.yml` · Environment: `pypi`
2. **Enable the publish job** — GitHub → repo *Settings* → *Secrets and variables* →
   *Actions* → *Variables* → set `PUBLISH_TO_PYPI = true`.
   (Until this is set, a tag still ships GitHub artifacts but skips the PyPI upload — safe.)

## Cutting a release

Work is version-stamped in the repo, so bump first if needed (must agree across all three):

- `pyproject.toml` → `version`
- `distil/__init__.py` → `__version__`
- `CITATION.cff` → `version`
- add a `## [X.Y.Z]` section to `CHANGELOG.md`

Then, from a clean `main`:

```bash
./scripts/release.sh            # preflight → push main → push tag → CI publishes → Homebrew sha
./scripts/release.sh --dry-run  # rehearse: prints every step, changes nothing
```

The driver fails fast before anything outward-facing: it checks you're on `main` with a
clean tree, that the version agrees across all surfaces, that the tag is free, that the
CHANGELOG has the entry, that tests pass, and that a **clean wheel install** (`--no-index`
into a throwaway venv) yields a working `distil bench`. Every network step asks first.

Pushing the tag triggers `release.yml`. Watch it:

```bash
gh run watch --repo dshakes/distil $(gh run list --repo dshakes/distil -w release -L1 --json databaseId -q '.[0].databaseId')
# or: https://github.com/dshakes/distil/actions/workflows/release.yml
```

## After CI is green

- **Verify PyPI:** `pipx install distil-llm && distil --version` → the new version.
- **Homebrew:** `release.sh` patches `packaging/homebrew/distil.rb` with the tag tarball's
  sha256. Copy that formula into the tap repo (`dshakes/homebrew-tap` → `Formula/distil.rb`)
  so `brew install dshakes/tap/distil` serves it.
- **GitHub release notes:** paste the `[X.Y.Z]` section of `CHANGELOG.md` into the release
  at `https://github.com/dshakes/distil/releases`.
- **Docs site** redeploys from `main` via `pages.yml` — confirm it's green.

## Escape hatch (Trusted Publishing not set up yet)

```bash
export UV_PUBLISH_TOKEN=pypi-…           # a token from pypi.org
./scripts/release.sh --local-publish     # uploads from this machine via `uv publish`
```

Only use this when CI is *not* publishing — uploading the same version twice fails (a PyPI
version is immutable and can never be re-uploaded).
