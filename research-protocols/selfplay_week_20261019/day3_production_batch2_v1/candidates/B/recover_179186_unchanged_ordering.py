"""Recover only unaffected original ordering definitions; failed MCs remain HOLD.
Original script is interpreted with a capture callback; its author call is not repeated.
"""
import runpy
import author_common as ac
original=ac.author
captured=[]
ac.author=lambda *args,**kwargs:captured.append((args,kwargs))
runpy.run_path(str(ac.OUT/'author_179186.py'))
ac.author=original
assert len(captured)==1
args,kwargs=captured[0]
packet=ac.load(ac.D/'packets/B/cc-train-episode-179186.json')
holds={s['slot_id']:'Original attempted definition has incorrect number of true options versus inherited target; mechanical generation gate failed before PLAN/public/private. Preserve original definition and record GENERATION_HOLD without changing its propositions or truth labels.' for s in packet['slots'][:2]}
ac.save(ac.OUT/'cc-train-episode-179186/FAILED_GENERATION_ATTEMPT.json',{
 'root_id':'cc-train-episode-179186','author':'/root/day3_public_design','recorded_at_utc':ac.now(),
 'exception':"AssertionError: ('179186', 'MC_BINDING', ['A', 'C', 'E'], 4)",
 'MC02_proposed_true_count':4,'MC02_inherited_target':5,
 'attempt_script_binding':ac.bind(ac.OUT/'author_179186.py'),
 'public_private_or_PLAN_existed_at_failure':False,'READ_MANIFEST_preserved':True,
 'resolution':'MC01 and MC02 GENERATION_HOLD. MC03 original SHORTFALL. Recover only the three original ordering definitions verbatim; no new anchor or Gold selected.'})
original(*args,**kwargs,recover_existing_manifest=True,generation_holds=holds)
