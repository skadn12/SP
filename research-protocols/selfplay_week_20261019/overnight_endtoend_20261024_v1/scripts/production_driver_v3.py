#!/usr/bin/env python3
"""Explicit engineering upgrade: v2 call guard + native evidence/source wrapper.

No author/reviewer prompt changes, model override or implicit scheduler start.
Original data, independent FAIL/HOLD, and original failure records are retained.
The separately authored source_unit_v3 module owns native-evidence recovery.
"""
import importlib.util
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
guard=module('oet1_driver_guard_v2',BASE/'scripts/production_driver_v2.py')
integrity=module('oet1_native_integrity_v2',BASE/'scripts/production_integrity_v2.py')
guard.integrity=integrity
driver=guard.driver
driver.control_path=BASE/'scripts/control_v3.py'

def main():
    unit=module('oet1_native_source_unit_v3',BASE/'scripts/source_unit_v3.py')
    driver.source_unit=lambda root,arm:unit.source_unit(driver,root,arm)
    guard.main()
if __name__=='__main__':main()
