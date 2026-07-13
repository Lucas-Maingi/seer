"""Face verification: ID portrait vs live selfie.

Detection (YuNet, 5 landmarks) → similarity-transform alignment to the
canonical ArcFace 112x112 template → ArcFace embedding → cosine similarity
against a threshold calibrated to a target false-match rate on a public
benchmark (LFW). The threshold is a *measured* operating point, not a vibe.
"""
