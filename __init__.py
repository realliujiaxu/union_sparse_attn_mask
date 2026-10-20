"""Reference implementation and tools for grouped sparse attention masks."""

from .union_sparse_torch import UnionSparseMetadata, build_union_sparse_metadata

__all__ = ["UnionSparseMetadata", "build_union_sparse_metadata"]
