import json
from pathlib import Path
A=Path(__file__).parent
selected=[('218519','MC2'),('225575','MC3'),('229686','MC3'),('218519','MC4'),('218519','ORDER1'),('218519','ORDER2'),('218519','ORDER3')]
lines=['# G组完整错题实例与两种展示原始作答','',
       '本文件逐字复制已封存题面和原始可见输出，不生成/补造思考。canonical为原题标签，display为该次Solver实际看到的标签；每次映射、Gold及最终预测分别列出。来源仅摘录支持分析的完整turn，原始完整对话路径另列。未重新调用模型。','']
for root,slot in selected:
    B=A/'reviews'/root; p=json.loads((B/'ERROR_REVIEW_PACKET.json').read_text()); r=next(x for x in p['records'] if x['candidate']['slot_id']==slot); c=r['candidate']
    review=json.loads((B/('ERROR_REVIEW_EFFECTIVE.json' if root=='218519' else 'ERROR_REVIEW.json')).read_text()); rr=next(x for x in review['records'] if x['slot_id']==slot)
    lines+=['## '+c['candidate_id'],'',c['question'],'','### canonical选项/节点','']
    for k,v in c.get('options',c.get('items')).items():lines.append(f'- {k}: {v}')
    lines+=['','canonical Gold: '+' '.join(c['gold']),'','完整来源: `'+p['source_binding']['path']+'`','']
    for x,rev in zip(r['responses'],rr['responses']):
        st=x['score']['strict']; lines+=['### '+x['request_id'],'','display → canonical: `'+json.dumps(x['presentation']['display_to_canonical'],ensure_ascii=False)+'`',
            '', 'display Gold: `'+ ' '.join(st['display_gold'])+'`；display prediction: `'+ ' '.join(st['display_prediction'] or [])+'`',
            '',f"官方 valid={st['valid']}, exact={st['exact']}；审核主分类={rev['primary_failure_code']}",
            '',rev['rationale'],'','原始可见输出（完整）:','','```json',x['raw_output'],'```','',
            '原始响应文件: `'+x['raw_binding']['path']+'`','']
    sources={z['turn_ref']:z['source_quote'] for rev in rr['responses'] for ev in rev['evidence'] for z in ev['source_checks']}
    lines+=['### 对照来源turn','']
    for ref,txt in sources.items(): lines+=['`'+ref+'`','',txt,'']
(A/'EXAMPLES_ZH.md').write_text('\n'.join(lines)+'\n')
print('Saved complete examples:',len(selected), 'questions;',sum(len(r) for r in lines),'characters')
