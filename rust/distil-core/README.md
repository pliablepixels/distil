# distil-core

PyO3 extension implementing the distil Tier-0 transforms and heuristic token
counter in Rust, with a pure-Python fallback in `distil.native`.

## Functions

| Python name | Rust pub fn | Description |
|---|---|---|
| `minify_json(text)` | `minify_json(&str) -> Option<String>` | Parse + re-emit JSON without whitespace; `None` for non-JSON |
| `collapse_runs(text)` | `collapse_runs(&str) -> String` | RLE-encode consecutive identical lines |
| `count_tokens(text, subword_factor=1.33)` | `count_tokens(&str, f64) -> usize` | Heuristic token count (`\w+\|[^\w\s]` × factor) |

All three functions match the Python semantics in `distil.compress.tier0` and
`distil.tokenizer` exactly.

## Build requirements

- Rust 1.70+ and Cargo
- [maturin](https://github.com/PyO3/maturin) (`uv tool install maturin`)
- Python 3.12+ with a project virtualenv

On macOS, Python 3.14 (system `python3`) ships as a Framework build.  The
`.cargo/config.toml` sets `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` so PyO3 0.29
accepts it.  `build.rs` locates the correct Python dylib for `cargo test`.
Maturin builds target the project virtualenv's Python interpreter automatically.

## Build and install (development)

```sh
# from the repo root
uv tool install maturin          # one-time
cd rust/distil-core
maturin develop --release        # builds and installs into .venv
```

Verify the Rust extension loaded:

```python
>>> import distil.native as n
>>> n.BACKEND
'rust'
>>> n.minify_json('{"a": 1}')
'{"a":1}'
>>> n.collapse_runs("x\nx\nx\n")
'x\n<<x3>>\n'
>>> n.count_tokens("hello world")
3
```

## Run Rust unit tests

```sh
cd rust/distil-core
cargo test
```

Output: `26 passed; 0 failed`

## Run Python parity tests

```sh
# from repo root
uv run python -m pytest tests/test_native.py -v
```

Output: `8 passed`

## Criterion benchmark numbers

Measured on Apple M-series (aarch64-apple-darwin, `--release`), 100 samples
each, on a ~30 KB synthetic log with repeated lines, prose, and JSON snippets:

| Function | Time (median) |
|---|---|
| `collapse_runs` | **2.48 µs** |
| `count_tokens` | **20.2 µs** |

`collapse_runs` is ~2.5 µs for a 30 KB input (~12 GB/s throughput).
`count_tokens` is ~20 µs, dominated by Unicode character-class scanning.

Run benchmarks yourself:

```sh
cd rust/distil-core
cargo bench
# HTML report: target/criterion/report/index.html
```

## Release build

```sh
cd rust/distil-core
cargo build --release
```

The cdylib at `target/release/libdistil_core.dylib` is the same artifact
maturin packages into the wheel.

## Python fallback

If the wheel is not installed, `distil.native` transparently falls back to
the pure-Python implementations and sets `BACKEND = "python"`.  No import
error is raised.  All `tests/test_native.py` assertions pass in both modes.
