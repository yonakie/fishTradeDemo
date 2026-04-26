"""Research layer — three parallel facets (fundamental / technical / sentimental)."""

from .fundamental import fundamental_node
from .sentimental import sentimental_node
from .technical import technical_node

__all__ = ["fundamental_node", "sentimental_node", "technical_node"]
