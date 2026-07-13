"""Tamper forensics on rectified documents.

Classical image-forensics signals (Error Level Analysis, high-pass noise
residuals) are computed deterministically and stacked as input channels for
a small CNN that both scores the document (tampered/clean) and localizes
*where* it thinks the edit happened. The classical channels inject priors a
small synthetic-data CNN would struggle to discover; the CNN supplies the
decision surface classical thresholds can't.
"""
