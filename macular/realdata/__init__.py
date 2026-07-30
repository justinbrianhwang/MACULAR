"""Adapters for PUBLIC, non-PHI real scanned-document datasets.

These convert public research datasets (FUNSD, XFUND) into MACULAR's Document
schema so the existing experiments (ocr_baseline, data_stats) run on genuine
scanned images instead of synthetic renders. This validates the OCR/layout
pipeline on real noise/layout without touching any patient data.

NONE of these datasets contain PHI. Real medical documents are out of scope
here and are gated behind the institutional governance path (proposal 14.6).
"""
