#!/usr/bin/env python3
"""Close the diagnostic at its frozen deadline, even if the UI disconnects."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
def main():
    clock=ROOT/'REVIEW_CLOCK.json'
    assert clock.exists(), 'review clock required; watchdog never starts sampling'
    with (ROOT/'WATCHDOG_STARTED.json').open('x') as f:
        json.dump({'pid':os.getpid(),'clock':str(clock),'purpose':'finalize once at frozen deadline; no inference or reviews'},f,indent=2)
    while True:
        state=json.loads(clock.read_text())
        if state['status']=='CLOSED': return
        left=(datetime.datetime.fromisoformat(state['deadline_utc'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if left<=0:
            subprocess.run([sys.executable,str(ROOT/'review_controller.py'),'finalize'],check=True)
            return
        time.sleep(min(30,left))
if __name__=='__main__': main()
