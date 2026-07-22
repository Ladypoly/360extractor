"""Occluder masking.

Static occluders live here (`geometric`), and what to do about them lives in `apply`.
Dynamic occluders -- people, passing cars, faces -- arrive in M3 as `ml`.
"""

from . import apply, geometric

__all__ = ["apply", "geometric"]
