"""monty-pod agent — a dumb render/inference executor.

Receives work from a control plane (poll, job-token auth), fetches media via presigned URLs,
applies a fully-resolved RenderSpec or runs a batched inference task, PUTs results back.
No editing decisions are made here: every number in the spec was decided upstream.
"""

__version__ = "0.1.0"
