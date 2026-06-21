#!/usr/bin/env node
/**
 * distil-proxy — thin Node.js launcher for the Distil compression proxy.
 *
 * This script is a launcher, not a reimplementation. Distil is a Python
 * package (PyPI: distil-llm). This wrapper locates or installs it, then
 * execs `distil proxy` with the arguments you pass.
 *
 * For pure-JS or TypeScript adoption you do not need to call this script
 * directly — just point your SDK's baseURL at the proxy once it's running:
 *
 *   npx @distil/proxy --port 8788 --upstream https://api.anthropic.com
 *   # then in your app:
 *   createAnthropic({ baseURL: 'http://127.0.0.1:8788' })
 *
 * Install: https://pypi.org/project/distil-llm/
 */

import { execSync, spawn } from "node:child_process";
import { existsSync } from "node:fs";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Try to run `cmd` silently; return true if it exits 0. */
function probe(cmd) {
  try {
    execSync(cmd, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/** Return the path to a working python3 binary, or null. */
function findPython() {
  for (const bin of ["python3", "python"]) {
    if (probe(`${bin} --version`)) return bin;
  }
  return null;
}

/** Return true if `distil` is importable by the given python binary. */
function distilInstalled(python) {
  return probe(`${python} -c "import distil"`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const python = findPython();

if (!python) {
  console.error(`
  distil-proxy: Python 3 not found.

  Distil is a Python package. Please install Python 3.11+ and try again:
    https://www.python.org/downloads/

  Then install Distil:
    pip install distil-llm

  Or use pipx for an isolated install:
    pipx install distil-llm
`);
  process.exit(1);
}

console.error(
  `[distil-proxy] This is a thin launcher around the Python 'distil-llm' package.`
);
console.error(`[distil-proxy] Python: ${python}`);

if (!distilInstalled(python)) {
  // Try pipx run first (zero-install, isolated).
  console.error(
    `[distil-proxy] 'distil' not found in your Python environment.`
  );
  console.error(`[distil-proxy] Attempting 'pipx run distil-llm proxy ...'`);

  if (probe("pipx --version")) {
    const pipxArgs = ["run", "distil-llm", "proxy", ...process.argv.slice(2)];
    const child = spawn("pipx", pipxArgs, { stdio: "inherit" });
    child.on("exit", (code) => process.exit(code ?? 0));
  } else {
    // Fall back to pip install --user.
    console.error(
      `[distil-proxy] pipx not found. Installing distil-llm via pip...`
    );
    try {
      execSync(`${python} -m pip install --user distil-llm`, {
        stdio: "inherit",
      });
      console.error(`[distil-proxy] Installation complete.`);
    } catch {
      console.error(`
  distil-proxy: Installation failed.

  Please install Distil manually:
    pip install distil-llm           # or
    pipx install distil-llm

  See: https://github.com/dshakes/distil
`);
      process.exit(1);
    }

    if (!distilInstalled(python)) {
      console.error(
        `[distil-proxy] Installation succeeded but 'distil' is still not importable.`
      );
      console.error(
        `[distil-proxy] Your pip --user site-packages may not be on PATH.`
      );
      console.error(
        `[distil-proxy] Try: export PATH="$(${python} -m site --user-base)/bin:$PATH"`
      );
      process.exit(1);
    }

    launchProxy(python);
  }
} else {
  launchProxy(python);
}

function launchProxy(python) {
  // Resolve the distil entry-point.  Prefer `distil proxy` on PATH; fall back
  // to `python -m distil proxy` which always works when the package is installed.
  const proxyArgs = ["proxy", ...process.argv.slice(2)];

  console.error(
    `[distil-proxy] Starting: ${python} -m distil ${proxyArgs.join(" ")}`
  );
  console.error(
    `[distil-proxy] Point your SDK's baseURL at http://127.0.0.1:8788 (default).`
  );

  const child = spawn(python, ["-m", "distil", ...proxyArgs], {
    stdio: "inherit",
  });

  child.on("exit", (code) => process.exit(code ?? 0));
}
