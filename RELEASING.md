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

## Release candidates and the soak window

The 1.10.0→1.11.3 day — six releases, each fixing the previous — is the failure mode
this policy exists to prevent. Fixes were correct in review and wrong under real use;
the missing ingredient was bake time, not more review.

**Any release that changes runtime behavior soaks as an rc first:**

1. Bump `pyproject.toml` to `X.Y.ZrcN` (PEP 440; the tag becomes `vX.Y.ZrcN`). Write the
   CHANGELOG entry under the **final** version `X.Y.Z` — the rc soaks for it.
2. `./scripts/release.sh` as usual. rc tags are handled automatically: the GitHub
   release is marked prerelease, Homebrew and the Docker image are skipped, and PyPI
   gets the rc (pip ignores prereleases unless asked — safe).
3. **Soak**: run the rc yourself on real work (`distil wrap` around your daily agent
   sessions) for **at least 3 days**. Beta users install with
   `pipx install --pip-args=--pre distil-llm`.
4. Any P0/P1 found → fix, cut `rcN+1`, restart the clock. A same-day follow-up final
   is the anti-pattern; a same-day follow-up rc is the system working.
5. Soak clean → bump to `X.Y.Z` on the same tree and release. That final is
   byte-identical code to the last rc plus only the version bump.

**Exempt from soak** (may go straight to a final): releases touching only docs, tests,
CI, or packaging metadata — nothing a running proxy/wrap/gateway executes.

## Cutting a release

The version is single-sourced from the installed package's metadata
(`importlib.metadata`, backed by `pyproject.toml`'s `version`) — `distil/__init__.py`'s
`__version__` is only a dev-mode fallback for running from a source checkout with
nothing installed, and doesn't need to be kept in sync. Bump before a release:

- `pyproject.toml` → `version`
- `CITATION.cff` → `version` and `date-released`
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
