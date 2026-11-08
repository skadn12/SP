"""Write an explicitly supplied human-readable agent adjudication; never infer PASS."""
from control import BASE, CHECKS, SCENES, read, write, bind, now, accept

def save_review(root, arm, scene, checks, findings, coverage_notes, decision='READY'):
    assert set(checks)==set(CHECKS)
    assert all(isinstance(v,dict) and v.get('status') in {'PASS','WARNING','FAIL','HOLD'} and v.get('reason') for v in checks.values())
    folder=BASE/'sources'/root/arm/scene.replace(':','_')
    source=read(folder/'CUMULATIVE_SOURCE.json'); draft=read(folder/'DRAFT.json')
    assert draft['author_id']!='/root'
    prior=read(draft['input_source_binding']['path'])
    assert source['turns'][:len(prior['turns'])]==prior['turns']
    assert source['turns'][len(prior['turns']):]==draft['turns']
    assert len(draft['turns'])==14
    value={'schema_version':'fbc1_independent_source_review_v1','root_id':root,'condition':arm,'scene_ref':scene,
      'reviewer_id':'/root','reviewer_type':'ai_agent','independent_of_source_author':True,
      'created_at_utc':now(),'source_binding':bind(folder/'CUMULATIVE_SOURCE.json'),'draft_binding':bind(folder/'DRAFT.json'),
      'full_source_read':True,'turn_refs_read':[t['turn_ref'] for t in source['turns']],
      'coverage_notes':coverage_notes,'blinding':'Nonblind coordinator review; root knows study goals but did not author dialogue. Same checks for both conditions.',
      'checks':checks,'findings':findings,'decision':decision}
    write(folder/'REVIEW.json',value)
    if decision=='READY': return accept(root,arm,scene)
    return {'decision':decision,'review':bind(folder/'REVIEW.json')}
