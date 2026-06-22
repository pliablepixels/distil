# Homebrew formula for Distil (PyPI: distil-llm).
#
# To use this formula:
#   brew tap dshakes/tap
#   brew install dshakes/tap/distil
#
# Or clone this file into your own tap at:
#   $(brew --repo)/Library/Taps/<yourname>/homebrew-tap/Formula/distil.rb
#
# sha256 is for the v0.1.0 source tarball. To recompute for a new version:
#   curl -sL https://github.com/dshakes/distil/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256

class Distil < Formula
  desc "Compression with a quality contract — context compression for LLM agentic runtimes"
  homepage "https://github.com/dshakes/distil"
  url "https://github.com/dshakes/distil/archive/refs/tags/v0.8.0.tar.gz"
  sha256 "6ca0963be513716d583fc1d8f59434fe283fe7076382878e33d539d9e6f38528"
  license "Apache-2.0"
  version "0.8.0"

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
