"""Document corner localization.

A single lightweight network regresses the four document corners as
sub-pixel keypoints via differentiable spatial-to-numerical transform
(DSNT / soft-argmax over predicted heatmaps). Corners — not boxes — because
the very next operation is a homography, and a homography needs four points.
"""
