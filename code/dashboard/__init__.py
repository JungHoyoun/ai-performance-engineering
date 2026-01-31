"""
GPU Performance Lab Dashboard

A sleek web dashboard for viewing benchmark results, LLM analysis,
and optimization insights.

Usage:
    python -m dashboard.api.server serve [--port 6970] [--data results.json]
"""

from .api.server import serve_dashboard

__all__ = ['serve_dashboard']

