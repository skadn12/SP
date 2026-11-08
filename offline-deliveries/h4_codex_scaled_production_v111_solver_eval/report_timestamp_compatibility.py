"""Execute frozen reporting with Python3.10 ISO timestamp compatibility only.

Original nanosecond author receipts and the frozen reporting script stay intact.
This wrapper changes only the precision used for timestamp comparisons, not any
evaluation, selection, scores, recorded source timestamps or generated content.
"""
from datetime import datetime
import importlib.util
from pathlib import Path
import re


class TimestampCompatibleDatetime(datetime):
    @classmethod
    def fromisoformat(cls, value):
        return super().fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', value))


if __name__ == '__main__':
    path = Path(__file__).with_name('report_production.py')
    spec = importlib.util.spec_from_file_location('frozen_production_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.datetime = TimestampCompatibleDatetime
    module.main()
