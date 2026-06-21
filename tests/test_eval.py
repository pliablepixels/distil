"""The certified compression frontier (proof pack)."""

from distil.eval import format_frontier, frontier, write_raw


def test_frontier_has_distil_certified_with_real_savings():
    report = frontier()
    dp = report.distil_point
    assert dp is not None
    assert dp.certified is True
    assert dp.equivalence == 1.0
    assert dp.savings > 0  # real lossless savings


def test_aggressive_truncation_falls_off_the_frontier():
    report = frontier()
    # the most aggressive truncation drops decisions -> not certified
    most = min(report.points, key=lambda p: int(p.label.split("@")[1]) if "@" in p.label else 9999)
    assert "truncate@" in most.label
    assert most.certified is False
    assert most.equivalence < 1.0


def test_certified_ceiling_is_positive_and_bounded():
    report = frontier()
    assert 0 < report.certified_ceiling < 1.0


def test_format_and_raw_output(tmp_path):
    report = frontier()
    text = format_frontier(report)
    assert "certified compression frontier" in text
    assert "distil" in text
    path = write_raw(report, str(tmp_path), "test")
    assert path.endswith("frontier-test.jsonl")
    lines = open(path).read().splitlines()
    assert len(lines) == len(report.points)
