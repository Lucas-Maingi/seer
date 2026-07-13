"""Field OCR for rectified documents.

A CRNN (conv feature extractor + BiLSTM + CTC) trained purely on the synth
engine's field crops. Because rectification puts every document into a
canonical frame, field locations are template constants — recognition runs
on known ROIs, not free-form text detection. MRZ lines are recognized with
the same model and then validated by ICAO 9303 check digits, which turns
OCR confidence into something you can actually trust.
"""
