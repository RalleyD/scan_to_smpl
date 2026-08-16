"""Evaluation: metrics and diagnostics.

Only the Tier 3 surface metrics are re-exported here. The Tier 2 diagnostics
(``ab_refit``, ``leave_one_view_out``, ``visualise``) are runnable scripts with heavy
imports and are imported from their modules directly.
"""

from scantosmpl.evaluation.surface_metrics import (
    ChamferReport,
    chamfer_report,
    point_to_surface_distances,
    tessellation_floor,
    vertex_to_point_distances,
)

__all__ = [
    "ChamferReport",
    "chamfer_report",
    "point_to_surface_distances",
    "tessellation_floor",
    "vertex_to_point_distances",
]
