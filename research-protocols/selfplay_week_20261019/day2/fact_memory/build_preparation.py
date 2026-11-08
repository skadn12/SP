#!/usr/bin/env python3
"""Prepare a source-bound, score-free memory pilot; never runs any model."""
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent.parent / 'day1'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return {'path': str(Path(path).resolve()), 'sha256': digest(path)}
def load(path): return json.loads(Path(path).read_text())
def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
def copy(path,dst):
    dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f: f.write(Path(path).read_bytes())

def main():
    if (HERE/'PREPARATION_FREEZE.json').exists(): raise SystemExit('Frozen preparation exists; create a new version instead.')
    now=datetime.now(timezone.utc).isoformat()
    roots=load(DAY1/'evaluation/ROOT_SELECTION.json')['memory_pilot']['root_ids']
    qs=load(DAY1/'diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    arms=['recent2','recent2_plus_prefix_memory','full_history']
    goals=[
      'Create one coherent follow-up conversation with at least one explicit small new action or decision. Let the visible context determine the subject. Distinguish reported facts, proposals, requests, and confirmed outcomes; do not assume unseen completion.',
      'Continue this branch with one useful next step or explicit revision of a visible unresolved matter. If none is visible, introduce a small ordinary collaborative task. Make any changed state explicit and preserve speaker attribution.',
      'Continue this branch with a concrete check, consequence, or follow-up to a visible matter. Resolve only what the dialogue explicitly establishes and leave genuinely pending outcomes pending.'
    ]
    protocol={
      'schema_version':'day2_fact_memory_pilot_v1','frozen_before_generation':True,'created_at_utc':now,
      'day1_protocol':ref(DAY1/'PROTOCOL_V1.json'),'fact_memory_schema':ref(DAY1/'diagnosis/FACT_MEMORY_SCHEMA.json'),
      'selection':{'rule':'Exactly day1 fixed first-six Train roots; no Solver scores','root_ids':roots},
      'arms':arms,'scenes_per_branch':3,'branch_count':18,'planned_scenes':54,
      'context':{'recent_unit':'last two complete accepted sessions in order; never two turns',
       'initial_prefix':'byte-identical full v125 historically accepted source, not a chopped older boundary',
       'scene1_paired':True,'later_scenes':'Each arm appends only its own previously accepted scene; later inputs are branch-dependent, not independent identical-context replications.',
       'shared_objectives':goals,'shared_objective_history_facts':False,
       'planner_visible':'Only its own arm rendered context and generic objective; no source-review constraints, sibling plans, questions, Gold, Solver feedback or evaluator-only full source.',
       'realizer_visible':'Same arm context plus that arm current planner output; no sibling output.',
       'speaker_policy':'Only existing speakers evidenced in this arm visible context; no inferred gender, age, profession, relationship or new speaking person.',
       'main_AB_context_identical':True},
      'generation':{'planner_calls_per_scene':1,'realizer_calls_per_scene':1,'review_calls_per_scene':1,
       'planner_max_output_tokens':1024,'realizer_max_output_tokens':4096,'memory_update_max_output_tokens':2048,
       'maximum_revisions_per_scene':1,'revision_max_output_tokens':4096,
       'planned_initial_generation_calls':108,'planned_review_calls':54,
       'maximum_revision_calls':54,'memory_updates':'Only memory arm, after accepted scenes1 and2: at most12 calls; no scene3 update needed.',
       'initial_memory':'Deterministic quote-index candidates prepared here; no model call, pending6 root-level semantic selection/validity reviews before memory-arm execution.',
       'initial_memory_review_calls_planned':6,'initial_memory_review_max_output_tokens':3072,
       'maintenance_review_calls_per_update':1,'maintenance_review_max_output_tokens':3072,
       'maximum_memory_update_calls':12,'maximum_memory_update_review_calls':12,
       'maximum_memory_maintenance_review_calls_including_initial':18,
       'memory_maintenance_budget_clarification':'Day2 operational clarification: initial1review/root and after each accepted scene1/2 one memory_update then one dedicated maintenance review. These reviews are separate memory-maintenance cost, not extra scene review. Scene3 has no future-context use, so no update/review is required; do not count18 updates. Failed or missing exact-hash memory review blocks next branch scene, retaining fixed denominator.',
       'actual_generation_calls':0,
       'backend_exact_id':None,'backend_temperature':None,'backend_seed':None,
       'unknown_parameters':'Log null with reason, never infer controls. Requested ceilings apply where exposed; unavailable controls must be logged before execution.'},
      'admission':{'length_turns':[12,20],
       'pass_rule':'Initial reviewed content must have zero critical/major conflicts or unsupported current-state claims, valid speakers, at least1 source-audited new action/decision, and duplication ratio<=0.25. Pure schema reserialization preserving identical dialogue is allowed after mechanical equivalence proof.',
       'reviewer_context':'Complete original accepted prefix plus this branch accepted prior scenes and candidate; blind arm labels when feasible; original-source quote citations mandatory for alleged conflicts.',
       'review_record_required':['root_id','branch_id','scene_index','candidate_sha256','accepted_prefix_sha256','decision','critical_major_error_count','new_progress_count','duplicate_turn_ratio','old_history_linkage_count','evidence_refs','cost','finished_at_utc'],
       'old_history_linkage':'Count only an explicit consistent connection to source before the initial last2 sessions; no credit for unsupported inventions or copying. Reviewer compares all plausible prior states.',
       'duplicate_turn_ratio':'Number of new turns whose normalized text has >=0.90 character similarity to a prior or earlier-new turn, divided by new turns. Boilerplate greetings <=5 words excluded from numerator and denominator; zero denominator fails.',
       'review_fail':'Stop this branch. A single revision may be retained as PENDING_REREVIEW exploration, never counted PASS or used as next prefix without separately authorized, separately versioned budget. No second review hidden in this pilot.',
       'fixed_denominator':'All18scenes/arm and all6roots remain; stopped-branch later scenes count missing/fail. Primary outcomes use initial review; revisions never replace failures.'},
      'quality_and_cost_adoption_frozen':{
       'primary_quality':'Root-equal mean accepted-initial-scenes/3; report question-free source consistency, full-branch completion and raw counts separately.',
       'required_memory_semantic_gate':'All6 initial memory candidates reviewed PASS at exact hash; every claim attributed or current-state verified, stale/retracted status explicit; pending/ambiguous facts excluded.',
       'minimum_memory_accepted_scenes':17,'maximum_memory_quality_shortfall_vs_full_scenes':1,
       'per_root_memory_vs_full':'Memory has at least as many accepted scenes as full on >=5/6 roots; remaining root loses at most1 scene.',
       'memory_vs_recent_if_recent_accepts_at_most16':'Memory accepts >=2 more scenes and improves >=2 roots; worsens at most1root. This is an operational effect threshold, not significance.',
       'memory_vs_recent_if_recent_accepts_at_least17':'No accepted-scene reduction vs recent; report ceiling and do not claim demonstrated memory-quality benefit.',
       'old_history_linkage':'Memory consistent-linkage scene count >= full minus1; report denominator18 and root detail, no cherry-picking successful scenes.',
       'cost_token_ratio_memory_over_full_max':0.80,'wall_clock_ratio_memory_over_full_max':1.10,
       'cost_scope':'All planner/realizer/review/revision calls, initial memory construction and review, memory maintenance and failed attempts. Report input/output tokens separately and total, with actual usage or explicitly model-tokenizer estimates.',
       'cost_comparison_gate':'For cost-based adoption both compared arms must complete all18scenes, every branch reaches scene3 and all usage/timing is recorded. Early stopping, missing usage or unavailable timing gives NO_COST_CONCLUSION; do not call failed-chain savings efficiency.',
       'quality_cost_logic':'Adopt memory only if EVERY semantic, quality, linkage and cost gate passes. Otherwise default full_history; do not adopt recent2 from this pilot.',
       'fallback':'full_history','six_roots_claim':'Operational small-pilot choice only; no statistical-significance or general-population claim.',
       'adjudication':'One source-first reviewer per scene returns fixed fields. Ambiguous evidence or missing required fields is HOLD, counted non-pass. No score-aware tie-break, third-arm inspection during generation, or post-outcome threshold edits.'},
      'memory_candidate_policy':{'status':'CANDIDATE_PENDING_SEMANTIC_REVIEW','coverage':'unknown; quote-index sampling is not a validated complete state memory',
       'selection':'From every session before the initial last2, keep the final utterance of the final two distinct speakers (or all if fewer), preserving source order. No question/Gold/score selection.',
       'claim_semantics':'Exact attributed utterance only; validity=uncertain for underlying current-state meaning. No paraphrased state inference.',
       'review_requirement':'A reviewer may select and explicitly classify valid state, supersession and unresolved issues in a separately frozen version; revalidate every quote, source hash and prefix chronology before use.',
       'projection':'Generation receives claim+validity+compact turn refs only. Full evidence/schema metadata remains evaluator-only; projection bytes and token costs are recorded.',
       'maintenance':'Only after accepted branch scenes; append own accepted facts, explicitly supersede old facts; never read future or sibling scenes. Each next-scene snapshot requires hash-bound semantic PASS; no unreviewed update enters context.'},
      'operational_clarifications':[{'topic':'Dedicated memory semantic-review budget','day1_modified':False,
       'reason':'Make source-first acceptance and post-scene update chronology executable without concealing maintenance calls in the one scene-review budget.',
       'frozen_before_any_pilot_generation':True}],
      'execution_state':'PREPARATION_ONLY_NO_MODEL_EXECUTION_NO_BRANCHES',
      'future_runtime_gate':'A controller must enforce branch receipts, semantic memory admission, hash lineage, isolated per-arm rendered input and complete actual cost. render_initial.py prepares only scene1; scenes2/3 reserved until accepted branch receipts exist.'
    }
    write(HERE/'PILOT_PROTOCOL.json',protocol)
    index=[]; plans=[]
    for root in roots:
      q=next(q for q in qs if q['root_id']==root)
      source=Path(q['source_path']); assert digest(source)==q['source_sha256']
      data=load(source); reviewpath=source.parent/'SOURCE_REVIEW.json'; review=load(reviewpath)
      assert review['source_sha256']==digest(source) and review['full_source_read'] is True and review['decision']=='READY'
      assert review['baseline_constraint_review']['status']=='PASS'
      assert set(t['turn_ref'] for t in data['turns'])<=set(review['turn_refs_read'])
      dst=HERE/'sources'/f'{root}.json'; copy(source,dst)
      gate=HERE/'source_gates'/f'{root}.json'; copy(reviewpath,gate)
      sessions=list(dict.fromkeys(t['session_ref'] for t in data['turns']))
      assert len(sessions)>=3
      old_sessions=sessions[:-2]; selected=[]
      for sess in old_sessions:
        rows=[(i,t) for i,t in enumerate(data['turns']) if t['session_ref']==sess]
        chosen=[]; seen=set()
        for i,t in reversed(rows):
          if t['speaker'] not in seen:
            chosen.append((i,t));seen.add(t['speaker'])
          if len(chosen)==2:break
        selected.extend(sorted(chosen))
      facts=[]
      for i,t in selected:
        quote=t['text']; assert quote
        facts.append({'fact_id':f'{root}:quote:{i:04d}','fact_type':'event','subject_ids':[t['speaker']],
          'claim':f"At source turn {i}, {t['speaker']} said: {quote}", 'validity':'uncertain',
          'valid_from_turn':i,'valid_until_turn':None,'supersedes_fact_ids':[],
          'evidence_refs':[{'source_id':data['source_id'],'source_sha256':digest(dst),'scene_id':t.get('scene_ref') or t['session_ref'],
          'turn_start':i,'turn_end':i,'speaker':t['speaker'],'quote':quote}],
          'uncertainty':'Exact historical attribution only. Underlying current validity, supersession, salience and coverage require source-first semantic review before use.'})
      memory={'document_type':'prefix_fact_memory','schema_version':'1.0','snapshot_id':f'd2:{root}:candidate:v000',
        'created_at':now,'root_id':root,'namespace':'train.expansion.fact_memory','audience':'expansion_planner_realizer',
        'source_id':data['source_id'],'source_sha256':digest(dst),'accepted_prefix':ref(dst),'prefix_last_turn':len(data['turns'])-1,
        'prefix_content_gate':ref(gate),'gold_or_question_conditioned':False,'future_turns_read':False,'parent_snapshot':None,
        'facts':facts,'known_limits':['CANDIDATE_PENDING_SEMANTIC_REVIEW; NOT ADMITTED FOR GENERATION',
         'Coverage unknown; deterministic older-session quotation index is not a validated complete state memory.',
         'Claims quote attributed historical speech; no current-state, identity or relationship inference.',
         'No source QA or Solver scores were used by the deterministic selection tool; preparation author had broader historical artifact access, so this is not external blind construction.']}
      mp=HERE/'memory_candidates'/f'{root}.json';write(mp,memory)
      rec={'root_id':root,'source':ref(dst),'historical_source':ref(source),'prefix_content_gate':ref(gate),
       'historical_review':ref(reviewpath),'turn_count':len(data['turns']),'session_count':len(sessions),
       'recent2_session_ids':sessions[-2:],'recent2_turn_count':sum(t['session_ref'] in sessions[-2:] for t in data['turns']),
       'memory_candidate':ref(mp),'memory_candidate_facts':len(facts),'memory_semantic_status':'PENDING','memory_coverage':'unknown'}
      index.append(rec)
      for arm in arms:
        for scene in range(1,4):
          plans.append({'root_id':root,'arm':arm,'branch_id':f'{root}:{arm}','scene_index':scene,
           'scene_id':f'D2:{root.split("-")[-1]}:{arm}:S{scene:02d}', 'high_level_objective':goals[scene-1],
           'status':'PENDING_MEMORY_SEMANTIC_GATE' if arm=='recent2_plus_prefix_memory' else ('RENDERABLE_INITIAL' if scene==1 else 'WAIT_PREVIOUS_BRANCH_ACCEPTANCE'),
           'planner_calls':1,'realizer_calls':1,'review_calls':1,'max_revision_calls':1})
    write(HERE/'SOURCE_BINDINGS.json',{'root_count':6,'roots':index})
    write(HERE/'PUBLIC_INPUT_PLAN.json',{'status':'PLANNED_ONLY','slots':plans,'planned_scenes':54,'actual_scenes':0})
    write(HERE/'MEMORY_REVIEW_QUEUE.json',{'status':'PENDING','candidates':[{'root_id':r['root_id'],'candidate':r['memory_candidate'],'decision':None,'coverage':'unknown'} for r in index],
       'reviewer_instructions':'Read exact full accepted source and candidate; verify each quote and attribution, distinguish report/plan/completion, current validity and supersession; a new version may improve selection. Source-bound mechanical validation is not semantic PASS. Bind final snapshot hash and every fact decision.'})
    print(json.dumps({'prepared_roots':6,'source_turns':sum(r['turn_count'] for r in index),'candidate_facts':sum(r['memory_candidate_facts'] for r in index),'planned_scenes':54,'actual_model_calls':0}))

if __name__=='__main__':main()
