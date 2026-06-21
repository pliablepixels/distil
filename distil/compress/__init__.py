"""Compression layer — risk-graded tiers.

Tier 0 (tier0): provably lossless, reconstructable transforms. Always safe.
Tier 1 (tier1): reversible digest behind a retrieval handle. Lossless in effect.
Aggressive (strategies): lossy baseline kept ONLY to prove the certification
                          gate actually rejects quality-degrading compression.
"""
