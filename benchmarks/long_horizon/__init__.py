"""Long-horizon ReAct coding agent benchmark for distil E7 (Phase 5 extension).

Exercises distil's relevance-gated reversible compression on genuinely long
conversations: a multi-turn ReAct agent that accumulates large peripheral context
(read_file output) over many turns, routed through the same compression proxy as
the aider SWE-bench eval.
"""
