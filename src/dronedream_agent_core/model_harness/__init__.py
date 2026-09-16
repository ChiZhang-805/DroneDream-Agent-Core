"""Model + Harness runtime boundaries, memory, design, graph, and model access.

Import concrete contracts from their defining submodule. Keeping this package
initializer dependency-free prevents cycles between the core contracts and the
executable Harness graph.
"""

__all__ = ["boundary", "design", "graph", "memory", "memory_projection", "model_port"]
