use criterion::{black_box, criterion_group, criterion_main, Criterion};
use distil_core::{collapse_runs, count_tokens};

/// Generate a realistic multi-KB log that contains runs of identical lines,
/// mixed content, and varied whitespace — representative of the kind of
/// telemetry / assistant output that distil processes.
fn make_realistic_log() -> String {
    let mut log = String::with_capacity(32 * 1024);

    // Preamble — varied lines
    for i in 0..50 {
        log.push_str(&format!(
            "2024-01-{:02} 12:{:02}:00 INFO  request_id=req-{:04} status=ok latency_ms={}\n",
            (i % 28) + 1,
            i % 60,
            i,
            10 + (i * 7) % 200
        ));
    }

    // Run of identical WARNING lines (triggers RLE)
    for _ in 0..20 {
        log.push_str("2024-01-15 12:30:00 WARN  rate limit approaching threshold 80%\n");
    }

    // More varied lines
    for i in 0..30 {
        log.push_str(&format!(
            "SELECT * FROM events WHERE tenant_id = {} AND created_at > '2024-01-01';\n",
            i
        ));
    }

    // Another run — empty separator lines
    for _ in 0..10 {
        log.push('\n');
    }

    // Dense prose section (many tokens)
    let prose = "The quick brown fox jumps over the lazy dog. \
                 Pack my box with five dozen liquor jugs. \
                 How vexingly quick daft zebras jump! \
                 The five boxing wizards jump quickly. ";
    for _ in 0..40 {
        log.push_str(prose);
        log.push('\n');
    }

    // JSON-ish lines (not collapsed, but tokenizer-heavy)
    for i in 0..20 {
        log.push_str(&format!(
            r#"{{"event":"page_view","user_id":{},"path":"/dashboard","ts":1700000000{:04}}}"#,
            i, i
        ));
        log.push('\n');
    }

    log
}

fn bench_collapse_runs(c: &mut Criterion) {
    let log = make_realistic_log();
    c.bench_function("collapse_runs", |b| {
        b.iter(|| collapse_runs(black_box(&log)))
    });
}

fn bench_count_tokens(c: &mut Criterion) {
    let log = make_realistic_log();
    c.bench_function("count_tokens", |b| {
        b.iter(|| count_tokens(black_box(&log), black_box(1.33)))
    });
}

criterion_group!(benches, bench_collapse_runs, bench_count_tokens);
criterion_main!(benches);
