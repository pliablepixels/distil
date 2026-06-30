# Homebrew formula for Distil (PyPI: distil-llm).
#
# To use this formula:
#   brew tap dshakes/tap
#   brew install dshakes/tap/distil
#
# Or clone this file into your own tap at:
#   $(brew --repo)/Library/Taps/<yourname>/homebrew-tap/Formula/distil.rb
#
# RELEASE STEP: after pushing the v1.0.0 tag, recompute and paste the sha256 below:
#   curl -sL https://github.com/dshakes/distil/archive/refs/tags/v1.0.0.tar.gz | shasum -a 256
# (the placeholder sha256 will make `brew install` fail loudly until filled — by design,
#  so a stale checksum can never silently install the wrong version.)

class Distil < Formula
  desc "Compression with a quality contract — context compression for LLM agentic runtimes"
  homepage "https://github.com/dshakes/distil"
  url "https://github.com/dshakes/distil/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "REPLACE_WITH_v1.0.0_TARBALL_SHA256"
  license "Apache-2.0"
  version "1.0.0"

  depends_on "python@3.12"

  def install
    # Create an isolated venv in libexec so Distil's stdlib-only package does
    # not pollute the user's Python environment.
    venv = libexec/"venv"
    system "python3.12", "-m", "venv", venv
    system "#{venv}/bin/pip", "install", "--no-deps", "."

    # Expose the `distil` entry-point as a shim in bin/.
    (bin/"distil").write <<~SH
      #!/bin/sh
      exec "#{venv}/bin/distil" "$@"
    SH
    chmod 0755, bin/"distil"
  end

  test do
    # Smoke-test: --version must exit 0 and print a version string.
    system bin/"distil", "--version"
  end
end
