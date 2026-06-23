# Homebrew formula for Distil (PyPI: distil-llm).
#
# To use this formula:
#   brew tap dshakes/tap
#   brew install dshakes/tap/distil
#
# Or clone this file into your own tap at:
#   $(brew --repo)/Library/Taps/<yourname>/homebrew-tap/Formula/distil.rb
#
# sha256 is for the v0.23.2 source tarball. To recompute for a new version:
#   curl -sL https://github.com/dshakes/distil/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256

class Distil < Formula
  desc "Compression with a quality contract — context compression for LLM agentic runtimes"
  homepage "https://github.com/dshakes/distil"
  url "https://github.com/dshakes/distil/archive/refs/tags/v0.23.2.tar.gz"
  sha256 "7f4dfa4831eae6805be1144beb2660a992e6a18c7f15c88758a0650949a86514"
  license "Apache-2.0"
  version "0.23.2"

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
