"""Shared test helpers for the AOI-PCB-SSD test suite.

Module-level functions (not fixtures) so they can be imported directly by test
modules.  Fixtures that are reused across multiple test files should also live
here — pytest loads conftest.py automatically for every collection run.
"""

import tensorflow as tf


def _cell(is_foreground: bool, offset: float = 0.0, cx: float = 0.5, cy: float = 0.5) -> list:
    """Build one encoded anchor row: ``[bg, ic, 8 offsets, cx, cy]``."""
    cls = [0.0, 1.0] if is_foreground else [1.0, 0.0]
    return cls + [offset] * 8 + [cx, cy]


def _batch(cells: list) -> tf.Tensor:
    """Wrap a list of anchor rows into a ``(1, n_boxes, 12)`` float32 tensor."""
    return tf.constant([cells], dtype=tf.float32)
