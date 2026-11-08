"""Index only artifacts from this approved four-root run."""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent
RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_full_pipeline_realizer_state_contract_v1_live_1')


def build():
    roots = json.loads((BASE/'AUTHORIZATION.json').read_text())['root_order']
    text = ['本索引只包含本次四条数据的实际输入输出。未产生的阶段标为未执行；模型原始响应与程序转换结果分别链接。', '']
    inventory = []
    stages = [
        ('完整冻结输入（原文/日期/Memory/反馈）', 'frozen/{root}/inputs.json'),
        ('初始Planner实际待发messages', 'frozen/{root}/planner_request.json'),
        ('Planner初始原始回答', 'expansion/roots/{root}/planner/output.json'),
        ('Planner初始候选校验', 'expansion/roots/{root}/planner/validation.json'),
        ('Gate初始原始回答', 'expansion/roots/{root}/semantic_gate/output.json'),
        ('Gate初始接收记录', 'expansion/roots/{root}/semantic_gate/validation.json'),
        ('Outline初始原始回答', 'expansion/roots/{root}/planner/outline_semantic_output.json'),
        ('Outline初始选择', 'expansion/roots/{root}/planner/outline_semantic_selection.json'),
        ('Planner补充原始回答', 'expansion/roots/{root}/planner/supplement_output.json'),
        ('补充Gate原始回答', 'expansion/roots/{root}/semantic_gate/output_supplement.json'),
        ('补充Outline原始回答', 'expansion/roots/{root}/planner/outline_semantic_output_supplement.json'),
        ('补充Outline选择', 'expansion/roots/{root}/planner/outline_semantic_selection_supplement.json'),
        ('状态来源接入前原计划', 'expansion/roots/{root}/planner/materialized_plan_before_state_handoff.json'),
        ('最终场景计划', 'expansion/roots/{root}/planner/materialized_plan.json'),
        ('下游实际来源证据', 'expansion/roots/{root}/realizer/source_state_context.json'),
        ('Realizer实际提示', 'expansion/roots/{root}/realizer/prompt.txt'),
        ('Realizer原始回答', 'expansion/execution/{root}/calls/{root}/realizer/response.txt'),
        ('Realizer场景ID绑定记录', 'expansion/roots/{root}/realizer/output_contract_batch_01.json'),
        ('Realizer程序转换输出', 'expansion/roots/{root}/realizer/output.json'),
        ('adherence原始回答', 'expansion/roots/{root}/realizer_adherence/output.json'),
        ('补写原始回答', 'expansion/execution/{root}/calls/{root}/continuation/response.txt'),
        ('复审原始回答', 'expansion/execution/{root}/calls/{root}/reaudit/response.txt'),
        ('正式晋升记录', 'expansion/roots/{root}/promotion/audit.json'),
        ('晋升后完整原文', 'expansion/roots/{root}/promotable_expanded_dialogue.md'),
        ('扩充结果', 'expansion/roots/{root}/outcome.json'),
        ('Miner/Writer完整原文', 'qa/roots/{root}/qa_transcript.txt'),
        ('Miner实际intent', 'qa/roots/{root}/intents.json'),
        ('Writer全部生成/repair记录', 'qa/roots/{root}/candidate_generation.json'),
        ('Writer保留题目', 'qa/roots/{root}/candidate_pool.json'),
        ('排序程序推导记录', 'qa/roots/{root}/ordering_integrity_audit.json'),
        ('QA结果', 'qa/roots/{root}/qa_summary.json'),
    ]
    for root in roots:
        text += [f'**{root}**', '', '| 阶段 | 文件 |', '|---|---|']
        for label, template in stages:
            p = RUN/template.format(root=root)
            exists = p.is_file()
            inventory.append({'root_id': root, 'stage': label, 'path': str(p), 'exists': exists})
            text.append(f'| {label} | [{p.name}]({p}) |' if exists else f'| {label} | 未产生 |')
        text.append('')
    text += [f'[完整调用及usage]({RUN}/complete_call_records.json) · [程序运行汇总]({RUN}/generation_summary.json)', '']
    (BASE/'reports/ARTIFACT_INDEX.md').write_text('\n'.join(text))
    (BASE/'reports/ARTIFACT_INDEX.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'indexed_existing_files': sum(r['exists'] for r in inventory)}))


if __name__ == '__main__':
    build()
