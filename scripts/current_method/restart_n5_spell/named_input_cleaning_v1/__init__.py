"""Isolated concrete-name cleaning for new expansion inputs.

Import the batch API from :mod:`.batch_clean` explicitly.  Keeping package
initialization side-effect free also makes ``python -m ...batch_clean``
deterministic and warning-free.
"""
