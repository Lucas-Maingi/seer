"""Synthetic specimen document engine.

Generates fictional Kenyan identity documents (national ID card and passport
data page) with complete ground truth for every downstream stage:

- 4 corner keypoints (localization)
- per-field text and quadrilaterals (OCR)
- portrait crop (face verification)
- tamper masks and labels (forensics)

Every rendered document carries a SPECIMEN watermark and belongs to a persona
that does not exist. No genuine security features are reproduced.
"""
