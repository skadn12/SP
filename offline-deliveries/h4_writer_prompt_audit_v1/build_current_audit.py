"""Read only frozen requests/code; emit an offline prompt audit here."""
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUN = BASE.parent/'h4_gate_downstream_v1/prepared_run'
CODE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1/scripts/current_method/restart_n5_spell')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = {x['path']:x['sha256'] for x in read(RUN/'freeze_manifest.json')['files']}
    code_records = []
    for rel in (
        'h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py',
        'h4_qa_mining_hybrid_generation_v1/candidate_contract.py',
        'h4_writer_partial_memory_v1/admission.py',
        'h4_writer_partial_memory_v1/runtime.py',
        'h4_writer_partial_memory_v1/repair_feedback.py',
        'h4_writer_partial_memory_v1/evidence_repair_feedback.py',
        'h4_ordering_grounded_options_v2/guidance.py',
    ):
        p = CODE/rel
        assert frozen[str(p)] == sha(p), str(p)
        code_records.append({'path':str(p),'sha256':sha(p),'matches_run_freeze':True})

    request_records = []
    for p in sorted((RUN/'qa/execution/calls').glob('*/*/request.json')):
        if 'intent-miner' in p.parent.name:
            continue
        root = p.parent.parent.name
        request = read(p)
        prompt_path = p.parent/'prompt.txt'
        prompt = prompt_path.read_text()
        raw_path = RUN/'qa/roots'/root/'raw/prompts'/(p.parent.name+'.txt')
        assert request['payload']['messages'] == [{'role':'user','content':prompt}]
        assert raw_path.read_text() == prompt
        prefix, source_and_tail = prompt.split('DIALOGUE:\n',1)
        transcript_path = RUN/'qa/roots'/root/'qa_transcript.txt'
        source = transcript_path.read_text().rstrip('\n')
        assert source_and_tail.startswith(source)
        primary = '-generator-' in p.parent.name
        if primary:
            assert source_and_tail.rstrip('\n') == source
        assert 'SCMS_QUALITY_GUIDANCE' not in prefix
        meta = read(p.parent/'response_metadata.json')
        assert meta['actual_model'] == 'qwen3.6-35b-a3b'
        assert meta['finish_reason'] == 'stop'
        request_records.append({
            'request_id':p.parent.name,'phase':'primary' if primary else 'repair',
            'prompt_path':str(prompt_path),'request_path':str(p),
            'prompt_sha256':sha(prompt_path),'request_sha256':sha(p),
            'transcript_path':str(transcript_path),'transcript_sha256':sha(transcript_path),
            'roles':['user'],'actual_payload_equals_saved_prompt':True,
            'whole_transcript_present':True,'raw_prompt_equals_dispatched_prompt':True,
            'actual_model':meta['actual_model'],'finish_reason':meta['finish_reason'],
            'prompt_chars':len(prompt),'prefix_chars':len(prefix),
            'ordering_prefix_chars':len(prefix[prefix.index('GROUNDED ORDERING'):]),
            'scms_quality_block_present':False,
        })
    assert Counter(r['phase'] for r in request_records) == {'primary':9,'repair':6}
    representative = RUN/'qa/execution/calls/cc-train-episode-259192/cc-train-episode-259192-contract_guided-generator-I002/prompt.txt'
    actual_lines = representative.read_text().splitlines()
    assert 'structurally grounded' in actual_lines[3]
    assert 'ordering only' in actual_lines[26]
    findings = [
        {
            'id':'SCMS_SEMANTIC_SCOPE_UNDERSPECIFIED','severity':'HIGH','status':'CONFIRMED_PROMPT_GAP',
            'finding':'SC/MS被明确要求结构落地：真实refs和groups关系；详细actor/state/context要求放在ordering-only段。缺少逐选项对照题干的主体、动作状态、目的关系，以及正确集all-and-only检查。',
            'source':str(representative),'line_start':2,'line_end':7,
            'additional_lines':[26,27,28,29,31],
            'not_claimed':'缺少显式条款不是授权编造，也不能证明仅靠补提示就能消除模型语义错误。',
            'observed_cases':['259192 I001 MS actor','112812 I002 MS actor','112812 I003 MS can/set'],
        },
        {
            'id':'NO_OPTION_LEVEL_SUPPORT_MAP','severity':'HIGH','status':'CONFIRMED_INTERFACE_GAP',
            'finding':'SC/MS输出只有整题evidence_refs/evidence_groups；示例仅一个SC，MS只描述A-E与list gold，没有各选项为何符合/不符合题干的可检查关联。排序却有逐节点绑定。',
            'source':str(representative),'line_start':6,'line_end':26,
            'code':str(CODE/'h4_qa_mining_hybrid_generation_v1/candidate_contract.py'),
            'code_lines':[95,99,114,128,164,166],
            'not_claimed':'新增支持表也仍是模型声明，程序最多保证ID和映射结构，不能自动认证支持关系。无需模型重复抄原文。',
        },
        {
            'id':'SOFT_INTENT_SCOPE_NOT_REALIGNED','severity':'MEDIUM','status':'CONFIRMED_BOUNDARY_GAP_WITH_OBSERVED_CASE',
            'finding':'允许更换证据组是合理的探索自由；但SC/MS没有强调更换事实后同步校准题干/变量，不得把同一人物的其他行动当作同一目的。原risk management题将Italian practice纳入Gold。',
            'source':str(representative),'line_start':3,'line_end':26,
            'not_claimed':'不是要恢复旧严格复制before/trigger/after；不能把intent当事实或禁止有据换题。',
        },
        {
            'id':'REPAIR_STRUCTURAL_ONLY','severity':'HIGH','status':'CONFIRMED_PIPELINE_BOUNDARY',
            'finding':'repair feedback明确是结构发现而非语义判断；已接收行冻结，故已接收的语义错题不会自动重新检查。月目标题repair仅修ID，问题/options/Gold不变后获接收。',
            'source':str(RUN/'qa/execution/calls/cc-train-episode-112812/cc-train-episode-112812-contract_guided-repair-I003/prompt.txt'),
            'line_start':191,'line_end':196,
            'not_claimed':'冻结已接收行是防连带重写的有意规则；应区分结构接受和内容资格，而不是撤销保护或无上限repair。',
        },
        {
            'id':'SCMS_SHORTAGE_PATH_UNAVAILABLE','severity':'MEDIUM','status':'CONFIRMED_DESIGN_RISK_NOT_CAUSAL_PROOF',
            'finding':'当前只允许ordering显式弃答；SC/MS仍以2+1为目标。后续validator要求MS至少2个Gold；缺少足够支持时没有MS显式证据不足分支。',
            'source':str(representative),'line_start':2,'line_end':7,
            'code':str(CODE/'h4_qa_multiroot_k3_t08_probe_v1/validator.py'),'code_lines':[52,55],
            'not_claimed':'配额是目标/上限和repair缺口，一次后允许不满额交付，不是整批强制四题；也不能证明模型B选项是为凑数而生成。',
        },
        {
            'id':'GROUP_SUBSET_PROMPT_CODE_MISMATCH','severity':'LOW','status':'CONFIRMED_INTERFACE_MISMATCH',
            'finding':'prompt要求groups为refs子集，normalizer明确允许存在的group ref不在flat refs中；后续validator亦不检查子集。此问题影响验收口径，但不是can/set或actor错误的充分原因。',
            'source':str(representative),'line_start':4,'line_end':4,
            'code':str(CODE/'h4_qa_mining_hybrid_generation_v1/candidate_contract.py'),'code_lines':[114,128],
        },
        {
            'id':'ORDERING_REQUIREMENTS_ALREADY_PRESENT','severity':'INFORMATION','status':'EXISTING_INSTRUCTIONS_VIOLATED_NOT_ABSENT',
            'finding':'排序已要求人物/状态真实、上下文完整、不得以提及/章节/常见流程推时间、单方向表示、无唯一顺序可弃答且不强修。既有错误不能全部解释为prompt没要求。',
            'source':str(representative),'line_start':27,'line_end':36,
        },
        {
            'id':'PROMPT_IMBALANCE','severity':'LOW','status':'MEASURED_CHARACTER_SHARE_NO_CAUSAL_CLAIM',
            'finding':'primary非原文部分约11.95k–12.01k字符，其中排序段约9.84k，占约82%；SC/MS主要是几句格式和结构规则。',
            'not_claimed':'字符份额不是注意力测量；未证明长prompt或顺序规则导致MS下降，且完整原文没有裁剪。',
        },
    ]
    rows = read(BASE.parent/'h4_gate_downstream_v1/reports/CONTENT_ALL_ROWS.json')
    current = {}
    for kind in ('single_choice','multi_select','ordering'):
        all_rows = [r for r in rows if r['question_type']==kind]
        final = [r for r in all_rows if r['in_final_pool']]
        current[kind] = {'raw_rows':len(all_rows),'program_final':len(final),'content_verdicts':dict(Counter(r['verdict'] for r in final))}
    manifest = {'status':'OFFLINE_AUDIT_COMPLETE','new_external_calls':0,'production_edits':0,'source_run_edits':0,
                'code_records':code_records,'request_records':request_records,
                'findings':findings,'current_counts':current,
                'scope':'Nine primary and six repair actual payloads; seven frozen source-code files plus targeted downstream validator; existing historical reports separately. Not a full-library audit.'}
    (BASE/'CURRENT_PROMPT_AUDIT.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    excerpts = ['# 实际请求的关键片段（附原文件行号）','',str(representative),'']
    for i,line in enumerate(actual_lines,1):
        if i<=36:
            excerpts.append(f'{i}: {line}')
    (BASE/'PROMPT_EXCERPTS.txt').write_text('\n'.join(excerpts)+'\n')
    print(json.dumps({'status':manifest['status'],'requests':len(request_records),'frozen_code':len(code_records),'counts':current,'external_calls':0},ensure_ascii=False))


if __name__ == '__main__':
    main()
