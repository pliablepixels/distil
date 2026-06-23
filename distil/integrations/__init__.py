"""Framework integrations — thin, in-process hooks for LiteLLM and LangChain.

Every framework already works with distil via the proxy (point its base URL at
``distil proxy`` — no code change). These modules add an *in-process* path for
teams that prefer not to run a sidecar: they compress the request before it leaves
the process, reusing the exact same reversible compression as the proxy.

Lazy by design: nothing here imports ``litellm`` or ``langchain`` at module load,
so importing :mod:`distil.integrations` never pulls a heavy optional dependency.
"""
