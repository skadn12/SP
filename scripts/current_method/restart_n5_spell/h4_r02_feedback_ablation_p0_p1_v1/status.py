"""Read-only stage accounting; update only this run's RUN_STATUS.md."""
import argparse,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from .freeze import DEFAULT,ARMS,ROOTS,load

def status(run):
    qs=[];ls=[];model_rows=[]
    for arm in ARMS:
        for rd in (run/arm/'qa/shards').glob('*/roots/*'):
            if (rd/'qa_generation_report.json').exists():qs.append({'arm':arm,**load(rd/'qa_generation_report.json')})
            ls.extend({'arm':arm,**load(p)} for p in (rd/'ledger/requests').glob('*.json'))
        sr=run/arm/'solver'
        if (sr/'preflight_report.json').exists():
            pre=load(sr/'preflight_report.json')
            for model in pre['models']:
                n=len(list((sr/model/'candidate_results').glob('*.json')))
                model_rows.append({'arm':arm,'model':model,'candidates':n,'expected_candidates':pre['candidate_count'],'requests_budget':pre['request_count_per_model'],'complete':(sr/model/'report.json').exists()})
    data={'qa_finished_root_arms':len(qs),'qa_ready_root_arms':sum(x['status']=='ready' for x in qs),'qa_expected_root_arms':12,'logical_calls_started':len(ls),'completions':sum(x['status']=='completed' for x in ls),'repair_calls':sum(x.get('request_metadata',{}).get('attempt')=='repair' for x in ls),'candidate_counts':{a:sum(x['valid_candidate_count'] for x in qs if x['arm']==a) for a in ARMS},'solver':model_rows}
    p=run/'audit/qa_transport_events.jsonl';events=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    outcomes=[e for e in events if e['event']=='logical_call_transport_outcome'];data['transport_failures']=sum(e['status']=='failed' for e in outcomes);data['timeouts']=sum(bool(e.get('timeout')) for e in outcomes)
    review=run/'diagnostics/blind_review/reviews';data['blind_reviews_finished']=len(list(review.glob('*.json')))
    complete=(run/'reports/p1_comparison.json').exists() and all(x['complete'] for x in model_rows) and len(model_rows)==6
    lines=['# R02 feedback P0–P1 状态','',f'更新时间：{datetime.now(timezone.utc).isoformat()}',f'根目录：{run}','','P0：PASS；6/6 配对输入，26/26 最终离线测试；三个 Solver 身份和原完整协议一致。',f"P1 QA：完成 {data['qa_finished_root_arms']}/12，ready {data['qa_ready_root_arms']}/12；逻辑调用 {len(ls)}/156，completion {data['completions']}，repair {data['repair_calls']}。",f"有效候选：off {data['candidate_counts']['feedback_off']}；on {data['candidate_counts']['feedback_on']}。",f"HTTP 技术失败 {data['transport_failures']}，超时 {data['timeouts']}。日志 audit/qa_live.log；原始 prompt/响应和账本在各臂 qa/shards/shard-00/roots。",'','P1 Solver：']
    lines += [f"- {x['arm']} / {x['model']}: {x['candidates']}/{x['expected_candidates']} 候选；请求预算 {x['requests_budget']}；{'完成' if x['complete'] else '待执行/执行中'}。" for x in model_rows] or ['尚未开始；实际预算将在 QA 冻结后精算。']
    lines+=['', 'Solver 总预算：' + str(sum(x['requests_budget'] for x in model_rows)) + ' 请求。',
            'Solver 日志：audit/p1_pipeline_live.log、audit/feedback_off_solver_live.log、audit/feedback_on_solver_live.log。',
            '进程记录：audit/p1_process_record.json（启动时 PID；当前状态以进程和最终退出记录为准）。',
            '阻碍：无待用户确认事项；等待原 Solver 完成时保留全部账本。' if not complete else '阻碍：无。评估、内容核查和报告均完成。',
            '',f"内容抽查：已存 {data['blind_reviews_finished']} 个盲化评审记录；最多 72 个选择。旧排序诊断 52/52 已完成。",'','来源局限：197635、254852 历史 seed 保留口径歧义，未作 Gold 认证。','无 Memory 调用、无 R02 Memory 写入、无训练、无 P2。','', 'P1 报告已生成；停止在 P1。' if complete else '继续执行已授权的 QA、三 Solver 与最终报告，当前未宣称 P1 完成。']
    (run/'RUN_STATUS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(data,ensure_ascii=False));return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);a=p.parse_args();status(a.run)
