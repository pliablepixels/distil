/// distil-core: Rust hot-path implementations of Tier-0 transforms and
/// heuristic token counting, exposed as a PyO3 extension module.
///
/// All three functions match the Python semantics exactly:
///   - `minify_json`  : parse + re-emit with serde_json (no whitespace)
///   - `collapse_runs`: regex-based RLE matching Python's `_RLE` pattern
///   - `count_tokens` : `\w+|[^\w\s]` segmentation × subword_factor, rounded
use pyo3::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// minify_json
// ──────────────────────────────────────────────────────────────────────────────

/// Re-encode JSON with no incidental whitespace.
///
/// Mirrors Python:
/// ```python
/// def minify_json(text: str) -> str | None:
///     s = text.strip()
///     if not (s[:1] in "{[" and s[-1:] in "}]"):
///         return None
///     try:
///         obj = json.loads(s)
///     except (ValueError, TypeError):
///         return None
///     return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
/// ```
pub fn minify_json(text: &str) -> Option<String> {
    let s = text.trim();
    let first = s.chars().next()?;
    let last = s.chars().next_back()?;
    if !matches!(first, '{' | '[') || !matches!(last, '}' | ']') {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(s).ok()?;
    // serde_json::to_string produces compact JSON without whitespace, and
    // preserves non-ASCII characters (ensure_ascii=False equivalent).
    serde_json::to_string(&value).ok()
}

// ──────────────────────────────────────────────────────────────────────────────
// collapse_runs
// ──────────────────────────────────────────────────────────────────────────────

/// Run-length-encode consecutive identical lines, reversibly.
///
/// Mirrors Python regex `^(.*)\n(?:\1\n)+` with MULTILINE flag.
///
/// The regex matches the first occurrence of a line followed by one or more
/// identical repetitions, where every line (including the last repetition)
/// ends with `\n`.  The full match is replaced by:
///   `{line}\n<<x{N}>>\n`
/// where N is the number of `\n` characters in the matched span.
///
/// This is a pure-Rust implementation that avoids a regex dependency by
/// scanning the text line-by-line, which is both faster and easier to
/// reason about.
pub fn collapse_runs(text: &str) -> String {
    // Split into lines preserving whether each has a trailing newline.
    // We work with byte slices to avoid extra allocation.
    //
    // Strategy: collect (line_content, has_trailing_newline) pairs, then
    // scan for runs of identical lines where every line in the run ends
    // with '\n'.  When a run of N>=2 is found (all with trailing \n),
    // emit `line\n<<xN>>\n` instead.

    // Build a list of (line_str, had_newline)
    let mut lines: Vec<(&str, bool)> = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        if let Some(pos) = remaining.find('\n') {
            lines.push((&remaining[..pos], true));
            remaining = &remaining[pos + 1..];
        } else {
            lines.push((remaining, false));
            remaining = "";
        }
    }

    let mut result = String::with_capacity(text.len());
    let mut i = 0;

    while i < lines.len() {
        let (line, has_nl) = lines[i];

        // Only start a run if this line ends with '\n' (had_newline=true).
        if !has_nl {
            // Last line with no trailing newline — can't be in a run.
            result.push_str(line);
            i += 1;
            continue;
        }

        // Count how many consecutive identical lines (all with '\n') follow.
        let mut run_len = 1usize;
        while i + run_len < lines.len() {
            let (next_line, next_nl) = lines[i + run_len];
            if next_nl && next_line == line {
                run_len += 1;
            } else {
                break;
            }
        }

        if run_len >= 2 {
            // Collapse: the match has `run_len` newlines.
            result.push_str(line);
            result.push('\n');
            result.push_str("<<x");
            let n_str = run_len.to_string();
            result.push_str(&n_str);
            result.push_str(">>\n");
            i += run_len;
        } else {
            result.push_str(line);
            result.push('\n');
            i += 1;
        }
    }

    result
}

// ──────────────────────────────────────────────────────────────────────────────
// count_tokens
// ──────────────────────────────────────────────────────────────────────────────

/// Count tokens using the heuristic segmenter.
///
/// Mirrors Python:
/// ```python
/// _PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
///
/// def count(self, text: str) -> int:
///     if not text:
///         return 0
///     pieces = _PIECE.findall(text)
///     return max(1, round(len(pieces) * self.subword_factor))
/// ```
///
/// The pattern `\w+|[^\w\s]` matches:
///   - one or more word characters (Unicode: letters, digits, underscore), OR
///   - a single non-word, non-whitespace character (punctuation, symbols)
///
/// We implement this without a regex crate by iterating over Unicode chars.
pub fn count_tokens(text: &str, subword_factor: f64) -> usize {
    if text.is_empty() {
        return 0;
    }

    let piece_count = count_pieces(text);
    let raw = (piece_count as f64) * subword_factor;
    // Python's `round()` uses "round half to even" (banker's rounding), but for
    // our purposes the values rarely fall on 0.5 boundaries.  We use the same
    // tie-to-even logic to be safe.
    let rounded = round_half_to_even(raw);
    rounded.max(1)
}

/// Count the number of pieces matching `\w+|[^\w\s]` in `text`.
fn count_pieces(text: &str) -> usize {
    let mut count = 0usize;
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];
        if is_word_char(c) {
            // Consume the entire \w+ run.
            count += 1;
            i += 1;
            while i < n && is_word_char(chars[i]) {
                i += 1;
            }
        } else if !c.is_whitespace() {
            // Non-word, non-whitespace: single-char piece.
            count += 1;
            i += 1;
        } else {
            // Whitespace: skip.
            i += 1;
        }
    }

    count
}

/// Returns true for `\w` characters: Unicode letters, digits, and `_`.
#[inline(always)]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Round a float to the nearest integer using "round half to even"
/// (the same default as Python's built-in `round()`).
fn round_half_to_even(x: f64) -> usize {
    let floor = x.floor() as i64;
    let frac = x - floor as f64;
    let rounded = if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1
    } else {
        // Exactly 0.5: round to even.
        if floor % 2 == 0 {
            floor
        } else {
            floor + 1
        }
    };
    rounded.max(0) as usize
}

// ──────────────────────────────────────────────────────────────────────────────
// PyO3 module
// ──────────────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "minify_json", signature = (text))]
fn py_minify_json(text: &str) -> Option<String> {
    minify_json(text)
}

#[pyfunction]
#[pyo3(name = "collapse_runs", signature = (text))]
fn py_collapse_runs(text: &str) -> String {
    collapse_runs(text)
}

#[pyfunction]
#[pyo3(name = "count_tokens", signature = (text, subword_factor = 1.33))]
fn py_count_tokens(text: &str, subword_factor: f64) -> usize {
    count_tokens(text, subword_factor)
}

#[pymodule]
fn distil_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_minify_json, m)?)?;
    m.add_function(wrap_pyfunction!(py_collapse_runs, m)?)?;
    m.add_function(wrap_pyfunction!(py_count_tokens, m)?)?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── minify_json ──────────────────────────────────────────────────────────

    #[test]
    fn minify_json_compact_object() {
        assert_eq!(
            minify_json(r#"{"a": 1, "b": 2}"#),
            Some(r#"{"a":1,"b":2}"#.to_string())
        );
    }

    #[test]
    fn minify_json_compact_array() {
        assert_eq!(minify_json("[1, 2, 3]"), Some("[1,2,3]".to_string()));
    }

    #[test]
    fn minify_json_strips_leading_trailing_whitespace() {
        assert_eq!(
            minify_json("  {\"key\": \"value\"}  "),
            Some(r#"{"key":"value"}"#.to_string())
        );
    }

    #[test]
    fn minify_json_unicode_preserved() {
        assert_eq!(
            minify_json(r#"{"unicode": "héllo"}"#),
            Some(r#"{"unicode":"héllo"}"#.to_string())
        );
    }

    #[test]
    fn minify_json_not_json() {
        assert_eq!(minify_json("not json"), None);
    }

    #[test]
    fn minify_json_invalid_json_object_like() {
        assert_eq!(minify_json("{not: valid}"), None);
    }

    #[test]
    fn minify_json_nested() {
        assert_eq!(
            minify_json(r#"{"a": {"b": [1, 2]}}"#),
            Some(r#"{"a":{"b":[1,2]}}"#.to_string())
        );
    }

    #[test]
    fn minify_json_empty_string() {
        assert_eq!(minify_json(""), None);
    }

    // ── collapse_runs ────────────────────────────────────────────────────────

    #[test]
    fn collapse_runs_no_repeats() {
        assert_eq!(collapse_runs("a\nb\nc\n"), "a\nb\nc\n");
    }

    #[test]
    fn collapse_runs_two_identical_lines() {
        // 2 identical lines (each with \n) → collapsed with <<x2>>
        assert_eq!(collapse_runs("a\na\n"), "a\n<<x2>>\n");
    }

    #[test]
    fn collapse_runs_three_identical_lines() {
        assert_eq!(collapse_runs("a\na\na\n"), "a\n<<x3>>\n");
    }

    #[test]
    fn collapse_runs_four_identical_lines() {
        assert_eq!(collapse_runs("a\na\na\na\n"), "a\n<<x4>>\n");
    }

    #[test]
    fn collapse_runs_mixed() {
        // "a\na\na\nb\nb\nb\nc" → note last 'c' has no trailing \n
        let input = "a\na\na\nb\nb\nb\nc";
        let expected = "a\n<<x3>>\nb\n<<x3>>\nc";
        assert_eq!(collapse_runs(input), expected);
    }

    #[test]
    fn collapse_runs_only_two_no_trailing_newline() {
        // "a\na" — last 'a' has no trailing \n so the run is only 1
        assert_eq!(collapse_runs("a\na"), "a\na");
    }

    #[test]
    fn collapse_runs_empty() {
        assert_eq!(collapse_runs(""), "");
    }

    #[test]
    fn collapse_runs_single_line() {
        assert_eq!(collapse_runs("hello\n"), "hello\n");
    }

    #[test]
    fn collapse_runs_empty_line_repeats() {
        // Three empty lines (each a "\n")
        assert_eq!(collapse_runs("\n\n\n"), "\n<<x3>>\n");
    }

    #[test]
    fn collapse_runs_interleaved() {
        let input = "x\nx\ny\nz\nz\nz\nz\n";
        let expected = "x\n<<x2>>\ny\nz\n<<x4>>\n";
        assert_eq!(collapse_runs(input), expected);
    }

    // ── count_tokens ─────────────────────────────────────────────────────────

    #[test]
    fn count_tokens_empty() {
        assert_eq!(count_tokens("", 1.33), 0);
    }

    #[test]
    fn count_tokens_hello_world() {
        // 2 word pieces → round(2 * 1.33) = round(2.66) = 3
        assert_eq!(count_tokens("hello world", 1.33), 3);
    }

    #[test]
    fn count_tokens_with_punctuation() {
        // "hello, world!" → pieces: hello , world ! = 4 → round(4 * 1.33) = round(5.32) = 5
        assert_eq!(count_tokens("hello, world!", 1.33), 5);
    }

    #[test]
    fn count_tokens_minimum_one() {
        // A single rare word → at least 1
        assert_eq!(count_tokens("a", 1.33), 1);
    }

    #[test]
    fn count_tokens_whitespace_only() {
        // Whitespace-only → 0 pieces → round(0 * 1.33) = 0 → max(1, 0) = 1
        // But wait: Python returns max(1, round(...)) = max(1, 0) = 1
        // However, if text is non-empty but has zero pieces... let's check:
        // The Python `if not text: return 0` guard triggers on empty string.
        // "   " is non-empty, so we proceed, find 0 pieces, return max(1, 0) = 1.
        assert_eq!(count_tokens("   ", 1.33), 1);
    }

    #[test]
    fn count_tokens_unicode_word() {
        // Unicode letters are \w — "héllo" is 1 piece
        // round(1 * 1.33) = round(1.33) = 1
        assert_eq!(count_tokens("héllo", 1.33), 1);
    }

    #[test]
    fn count_tokens_custom_factor() {
        // "foo bar baz qux" = 4 pieces × 2.0 = 8
        assert_eq!(count_tokens("foo bar baz qux", 2.0), 8);
    }

    #[test]
    fn count_tokens_parity_check() {
        // Matches expected Python output for known inputs
        assert_eq!(count_tokens("foo bar baz qux", 1.33), 5);
    }
}
