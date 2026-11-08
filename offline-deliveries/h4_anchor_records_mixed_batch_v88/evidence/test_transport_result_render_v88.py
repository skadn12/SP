"""Test completion rendering in synthetic fixtures; never read Solver outputs."""
import ast
from pathlib import Path

HOME=Path(__file__).resolve().parents[1]
source=HOME/'evidence/verify_completion_offline_v88.py'
tree=ast.parse(source.read_text())
block=next(node for node in tree.body if isinstance(node,ast.With))
statements=[]
for statement in block.body:
    statements.append(statement)
    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='state' for t in statement.targets):
        break
else:raise AssertionError('synthetic_fixture_state_anchor_missing')
checks=ast.parse("""
state['audits']['GENERATION_AUDIT']['totals'].update(successful_generation_http_calls=7,failed_generation_http_calls=1)
state['audits']['GENERATION_AUDIT']['transport_failures']=[{'root_id':'synthetic-source','round_id':'supplement','http_status':503,'error_type':'ComparisonTransportError'}]
rendered=completion.results_text(state)
assert '成功响应 7 次、传输失败 1 次' in rendered
assert 'HTTP 503' in rendered and '不计作题意或反例内容拒绝' in rendered
names=('ORDERING_EXCLUSION_SUBSTANTIVE_DIAGNOSIS.json','ORDERING_EXCLUSION_SUBSTANTIVE_DIAGNOSIS.md','TRANSPORT_AUDIT_RECOVERY_v88.json')
for filename in names:
    (evidence/filename).write_text('Synthetic index fixture only.')
index=completion.index_text(state,home)
assert all(filename in index for filename in names)
print('PASS synthetic failed-transport result disclosure and artifact-index coverage; no actual Solver artifacts read')
""").body
block.body=statements+checks
tree.body=tree.body[:tree.body.index(block)+1]
exec(compile(ast.fix_missing_locations(tree),'synthetic_transport_renderer','exec'),{'__file__':str(source)})
