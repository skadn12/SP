"""Source-bound observations for Day 2. No inference, Gold edits, or cause diagnosis.

All inference code remains frozen. Outputs are exclusive files in a new output
folder. If scores are unavailable, write an explicit WAITING package.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, itertools, json, math, pathlib, re, statistics
from datetime import datetime,timezone
HERE=pathlib.Path(__file__).resolve().parent
DAY2=HERE.parent
DEFAULT_DIRECT=DAY2/'solver/direct'
DEFAULT_EVIDENCE=DAY2/'solver/evidence'
BOX=re.compile(r'\s*<box>\s*(.*?)\s*</box>\s*\Z',re.S)


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def bind(p):
    p=pathlib.Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(pathlib.Path(p).read_text())
def require(c,m):
    if not c:raise RuntimeError(m)
def save(p,obj):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def verify(rows):
    for b in rows:
        p=pathlib.Path(b['path']);require(p.is_file() and sha(p)==b['sha256'] and p.stat().st_size==b['bytes'],'BOUND_FILE_CHANGED:'+str(p))
def mean(v):return statistics.mean(v) if v else None
def ratio(a,b):return a/b if b else None
def scene(ref):return ref.rsplit(':T',1)[0] if isinstance(ref,str) and ':T' in ref else None

def citations(value):
    """Collect all declared author citation locations; retain no semantic verdict."""
    rows=[]
    def visit(v):
        if isinstance(v,dict):
            if isinstance(v.get('turn_ref'),str):rows.append({k:v.get(k) for k in ['turn_ref','speaker','quote']})
            for k,x in v.items():
                if isinstance(x,(dict,list)):visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    visit(value);out=[];used=set()
    for r in rows:
        key=tuple(r.get(k) for k in ['turn_ref','speaker','quote'])
        if key not in used:out.append(r);used.add(key)
    return out

def author_anchor(q,label):
    evidence=q.get('option_evidence' if q['question_type']=='multi_select' else 'item_evidence',{}).get(label,{})
    anchor=(q.get('order_anchor_spans') or {}).get(label)
    if not anchor:anchor=evidence.get('order_anchor') or evidence.get('primary_citation') or evidence
    return evidence,anchor,citations(evidence)+[x for x in citations(anchor) if x not in citations(evidence)]

def author_primary_ref(q,label):
    _,a,c=author_anchor(q,label)
    return a.get('turn_ref') if isinstance(a,dict) and a.get('turn_ref') else next((r['turn_ref'] for r in c if r.get('turn_ref')),None)

def validate_citation(entry,source,q,display_mapping,canonical_mapping,request_id,ordinal):
    """Mechanical provenance only. A different valid location is not a semantic error."""
    turns={t['turn_ref']:t for t in source['turns']};positions={t['turn_ref']:i for i,t in enumerate(source['turns'])}
    display=entry.get('label');content=display_mapping.get(display);inverse={c:l for l,c in canonical_mapping.items()};label=inverse.get(content)
    ref=entry.get('turn_ref');quote=entry.get('quote');turn=turns.get(ref);status=entry.get('status')
    empty=(ref in ('',None) and quote in ('',None))
    quote_exact=bool(turn and isinstance(quote,str) and quote and quote in turn['text'])
    occurrences=[m.start() for m in re.finditer(re.escape(quote),turn['text'])] if quote_exact else []
    all_quote_refs=[t['turn_ref'] for t in source['turns'] if isinstance(quote,str) and quote and quote in t['text']]
    base={'request_id':request_id,'ordinal':ordinal,'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],
      'record_kind':'MISSING_MODEL_ENTRY' if entry.get('_missing_model_entry') else 'MODEL_ENTRY','display_label':display,'canonical_label':label,'content_id':content,'model_status':status,'turn_ref':ref,'quote':quote,
      'turn_exists':turn is not None,'quote_exact_in_cited_turn':quote_exact,'quote_offsets':occurrences,
      'quote_occurrence_turn_refs_in_complete_source':all_quote_refs,'quote_unique_across_source':len(all_quote_refs)==1,
      'source_speaker_at_citation':turn['speaker'] if turn else None,'source_scene_at_citation':scene(ref) if turn else None,
      'source_turn_index':positions.get(ref),'model_speaker_field':entry.get('speaker'),'model_speaker_claim_status':'NOT_REQUESTED_BY_SCHEMA' if 'speaker' not in entry else 'PROVIDED',
      'empty_citation':empty,'empty_citation_status_compatible':empty and status=='not_supported',
      'citation_mechanically_valid':quote_exact or (empty and status=='not_supported'),
      'semantic_judgment':'NOT_DETERMINED_BY_LOCATION_OR_STRING_CHECK'}
    if label is None:
        base.update(location_relation_to_author='UNKNOWN_DISPLAY_LABEL',author_citations=[],model_status_agrees_with_author=None);return base
    e,anchor,auth=author_anchor(q,label);authrefs={c['turn_ref'] for c in auth if c.get('turn_ref')}
    authspeakers={c.get('speaker') for c in auth if c.get('speaker')};authscenes={scene(r) for r in authrefs}
    authored_status={'supports':'supported','contradicts':'contradicted','not_supported':'not_supported','unsupported':'not_supported','ambiguous':'ambiguous'}.get(e.get('relation'))
    if q['question_type']=='ordering' and authored_status is None and e.get('supported') is True:authored_status='supported'
    if empty:location='NO_CITATION'
    elif turn is None:location='TURN_REF_NOT_IN_SOURCE'
    elif ref in authrefs:location='SAME_TURN_AS_AUTHOR'
    elif scene(ref) in authscenes and turn['speaker'] in authspeakers:location='DIFFERENT_TURN_SAME_SCENE_AND_SPEAKER'
    elif scene(ref) in authscenes:location='DIFFERENT_TURN_SAME_SCENE'
    elif turn['speaker'] in authspeakers:location='DIFFERENT_SCENE_SAME_SPEAKER'
    else:location='DIFFERENT_SCENE_AND_SPEAKER'
    author_checks=[]
    for c in auth:
        t=turns.get(c.get('turn_ref'));text=c.get('quote')
        author_checks.append({**c,'turn_exists':t is not None,'quote_exact_in_cited_turn':bool(t and isinstance(text,str) and text and text in t['text']),
          'speaker_matches_source':bool(t and c.get('speaker')==t['speaker']) if c.get('speaker') else None})
    base.update(author_primary_ref=author_primary_ref(q,label),author_citations=author_checks,author_status=authored_status,
      model_status_agrees_with_author=(status==authored_status) if authored_status and status in {'supported','contradicted','not_supported','ambiguous'} else None,
      location_relation_to_author=location,same_speaker_as_any_author_citation=(turn['speaker'] in authspeakers) if turn and authspeakers else None,
      exact_quote_equals_any_author_quote=any(quote==c.get('quote') for c in auth if isinstance(quote,str) and quote),
      model_speaker_matches_source=(entry.get('speaker')==turn['speaker']) if turn and 'speaker' in entry else None)
    return base

def parse_response(raw,row):
    if raw.get('status')!='RETURNED':return None,'REQUEST_'+str(raw.get('status'))
    if raw.get('metadata',{}).get('finish_reason') not in ('stop','eos'):return None,'INCOMPLETE_COMPLETION'
    text=raw.get('raw_output');field='options' if row['question_type']=='multi_select' else 'items';allowed=set(row['display'][field])
    try:
        if row['branch']=='evidence':
            obj=json.loads(text);answer=obj['answer']
            require(isinstance(answer,list) and all(isinstance(s,str) for s in answer),'ANSWER_MUST_BE_STRING_ARRAY')
        else:
            m=BOX.fullmatch(text or '');require(m is not None,'EXACTLY_ONE_BOX_REQUIRED')
            answer=[s.strip().upper() for s in m.group(1).split(',')]
        require(answer and len(answer)==len(set(answer)) and set(answer)<=allowed,'INVALID_OR_DUPLICATE_LABEL')
        if row['question_type']=='ordering':require(len(answer)==len(allowed),'INCOMPLETE_ORDER')
        return answer,None
    except Exception as e:return None,str(e)

def permutation_observation(records,qtype):
    groups=collections.defaultdict(list)
    for r in records:
        if r['valid']:groups[r['variant_id']].append(tuple(r['canonical_prediction']))
    within_total=within_different=0;groupstats=[]
    for vid,values in sorted(groups.items()):
        pairs=list(itertools.combinations(values,2));different=sum(a!=b for a,b in pairs)
        within_total+=len(pairs);within_different+=different
        groupstats.append({'variant_id':vid,'valid_samples':len(values),'distinct_canonical_answers':len(set(values)),
           'comparable_sample_pairs':len(pairs),'different_sample_pairs':different})
    tv=[]
    for left,right in itertools.combinations(sorted(groups),2):
        l=collections.Counter(groups[left]);r=collections.Counter(groups[right]);nl=sum(l.values());nr=sum(r.values())
        keys=set(l)|set(r);value=sum(abs(l[k]/nl-r[k]/nr) for k in keys)/2
        tv.append({'left_variant':left,'right_variant':right,'empirical_total_variation':value,'left_valid_samples':nl,'right_valid_samples':nr})
    pooled=[x for values in groups.values() for x in values];cnt=collections.Counter(pooled)
    return {'within_mapping_sample_pairs':within_total,'within_mapping_different_pairs':within_different,
      'within_mapping_pair_disagreement_valid_only':ratio(within_different,within_total),
      'within_mapping_variation_is_estimable':within_total>0,'within_mapping_details':groupstats,
      'between_mapping_distribution_tv_mean_valid_only':mean([r['empirical_total_variation'] for r in tv]),'between_mapping_details':tv,
      'pooled_distinct_canonical_answers':len(cnt),'pooled_modal_fraction_valid_only':max(cnt.values())/len(pooled) if pooled else None,
      'interpretation':'Descriptive variation only. Between-mapping differences mix display changes with independently seeded stochastic sampling. No causal position-sensitivity claim.',
      'ordering_single_sample_limitation':qtype=='ordering'}

def load_run(run):
    manifest=read(run/'manifest.json');scores=read(run/'SCORES.json');freeze=read(run/'RAW_RESPONSE_FREEZE.json')
    require(scores['raw_response_freeze_binding']['sha256']==sha(run/'RAW_RESPONSE_FREEZE.json'),'SCORE_FREEZE_DISAGREEMENT')
    verify(freeze['files']);verify([freeze['manifest_binding']]);verify([manifest['private_answer_binding']])
    require(len(freeze['files'])==len(manifest['requests'])==scores['denominator_calls'],'FIXED_DENOMINATOR_MISMATCH')
    private={q['candidate_id']:q for q in read(manifest['private_answer_binding']['path'])['questions']}
    maps=read(run/'private/DISPLAY_MAPPINGS.json');scoreby={r['request_id']:r for r in scores['requests']}
    observations=[];citations_out=[];bindings=[bind(run/'manifest.json'),bind(run/'SCORES.json'),bind(run/'RAW_RESPONSE_FREEZE.json'),bind(run/'private/DISPLAY_MAPPINGS.json'),manifest['private_answer_binding']]
    questions=[];rawrecords={};sourcecache={};errors=[]
    for row in manifest['requests']:
        rid=row['request_id'];raw=read(run/'responses'/f'{rid}.json');rawrecords[rid]=raw;bindings.append(bind(run/'responses'/f'{rid}.json'))
        q=private[row['candidate_id']]
        if q['root_id'] not in sourcecache:
            sourcepath=run/'sources'/f"{q['root_id']}.json";require(sha(sourcepath)==q['source_sha256'],'SOURCE_VERSION_CHANGED')
            sourcecache[q['root_id']]=read(sourcepath);bindings.append(bind(sourcepath))
        source=sourcecache[q['root_id']];promptpath=run/'prompts'/f'{rid}.txt'
        require(sha(promptpath)==row['prompt_sha256'],'PROMPT_VERSION_CHANGED')
        prompt=promptpath.read_text();covered=sum(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns'])
        coverage=covered/len(source['turns']);require(coverage==1.0,'INCOMPLETE_SOURCE_IN_PROMPT')
        cmap=maps['public_to_canonical'][q['candidate_id']];inverse={v:k for k,v in cmap.items()};display=maps['requests'][rid]['display_to_content']
        gold=q['gold'];goldcontent=[cmap[x] for x in gold];answer,parse_error=parse_response(raw,row)
        prediction=[display[x] for x in answer] if answer else None
        if prediction and q['question_type']=='multi_select':prediction=sorted(prediction)
        score=scoreby[rid];runtime_valid=score.get('valid') is True
        independently_parseable=answer is not None
        if independently_parseable and not runtime_valid and row['branch']=='direct':errors.append({'request_id':rid,'error':'RUNTIME_INVALID_BUT_RAW_PARSEABLE'})
        valid=runtime_valid and independently_parseable
        exact=valid and ((set(prediction)==set(goldcontent)) if q['question_type']=='multi_select' else prediction==goldcontent)
        if bool(score.get('exact'))!=bool(exact):errors.append({'request_id':rid,'error':'EXACT_SCORE_RECOMPUTATION_DISAGREES'})
        obs={'request_id':rid,'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],'variant_id':row['variant_id'],
          'sample_slot':row['slot'],'valid':valid,'independently_parseable':independently_parseable,'runtime_valid':runtime_valid,'parse_error':parse_error,
          'canonical_prediction':prediction,'canonical_label_prediction':[inverse[x] for x in prediction] if prediction else None,'original_gold_labels':gold,
          'gold_count':len(gold),'exact':bool(exact),'selected_count':len(prediction) if valid else None,'source_coverage':coverage,
          'response_committed_at_utc':raw.get('response_committed_at_utc'),'raw_binding':bind(run/'responses'/f'{rid}.json')}
        if q['question_type']=='multi_select':
            pred=set(prediction or []);g=set(goldcontent)
            obs.update(tp=len(pred&g) if valid else None,fp=len(pred-g) if valid else None,fn=len(g-pred) if valid else None,
              false_positive_labels=[inverse[x] for x in pred-g] if valid else None,false_negative_labels=[inverse[x] for x in g-pred] if valid else None,
              jaccard=len(pred&g)/len(pred|g) if valid else None,overselected=(len(pred)>len(g)) if valid else None,
              selected_all=(len(pred)==len(cmap)) if valid else None)
        else:
            pairs=[];rank={v:i for i,v in enumerate(prediction or [])}
            for left,right in itertools.combinations(gold,2):
                lref=author_primary_ref(q,left);rref=author_primary_ref(q,right);kind='UNKNOWN_ANCHOR_SCENE' if not scene(lref) or not scene(rref) else 'WITHIN_SCENE' if scene(lref)==scene(rref) else 'ACROSS_SCENES'
                pairs.append({'earlier_gold_label':left,'later_gold_label':right,'correct':bool(valid and rank[cmap[left]]<rank[cmap[right]]),
                  'valid_prediction':valid,'kind':kind,'earlier_author_turn_ref':lref,'later_author_turn_ref':rref})
            obs.update(ordering_pairs=pairs,pair_correct=sum(x['correct'] for x in pairs),pair_denominator=len(pairs),
              pair_accuracy_fixed_denominator=sum(x['correct'] for x in pairs)/len(pairs),pair_accuracy_valid_only=sum(x['correct'] for x in pairs)/len(pairs) if valid else None)
        observations.append(obs)
        if row['branch']=='evidence':
            if q['root_id'] not in sourcecache:
                path=run/'sources'/f"{q['root_id']}.json";require(sha(path)==q['source_sha256'],'SOURCE_VERSION_CHANGED');sourcecache[q['root_id']]=read(path);bindings.append(bind(path))
            try:entries=json.loads(raw.get('raw_output') or '{}').get('evidence',[])
            except Exception:entries=[]
            if not isinstance(entries,list):entries=[]
            entries=[e for e in entries if isinstance(e,dict)]
            field='options' if row['question_type']=='multi_select' else 'items'
            present={e.get('label') for e in entries if isinstance(e.get('label'),str)}
            entries.extend({'label':label,'status':None,'turn_ref':'','quote':'','_missing_model_entry':True} for label in row['display'][field] if label not in present)
            for i,entry in enumerate(entries):
                if isinstance(entry,dict):citations_out.append(validate_citation(entry,sourcecache[q['root_id']],q,display,cmap,rid,i))
    for qid,q in private.items():
        records=[r for r in observations if r['candidate_id']==qid]
        if not records:continue
        perms=permutation_observation(records,q['question_type']);valid=[r for r in records if r['valid']]
        question={'candidate_id':qid,'root_id':q['root_id'],'question_type':q['question_type'],'gold_count':len(q['gold']),
          'planned_calls':len(records),'valid_calls':len(valid),'exact_calls':sum(r['exact'] for r in records),'exact_rate':sum(r['exact'] for r in records)/len(records),
          'permutation_observations':perms,'source_coverage':1.0,'stratum':next((x['stratum'] for x in scores['questions'] if x['candidate_id']==qid),None)}
        if q['question_type']=='multi_select':
            question.update(tp=sum(r['tp'] for r in valid),fp=sum(r['fp'] for r in valid),fn=sum(r['fn'] for r in valid),
              mean_jaccard_valid_only=mean([r['jaccard'] for r in valid]),overselection_rate_valid_only=mean([r['overselected'] for r in valid]),
              all_selection_rate_valid_only=mean([r['selected_all'] for r in valid]),mean_selected_count_valid_only=mean([r['selected_count'] for r in valid]))
        else:
            pairs=[p for r in records for p in r['ordering_pairs']];question.update(ordering_pair_correct=sum(p['correct'] for p in pairs),ordering_pair_denominator=len(pairs),
              pair_accuracy_fixed_denominator=sum(p['correct'] for p in pairs)/len(pairs),pair_accuracy_valid_only=mean([r['pair_accuracy_valid_only'] for r in valid]))
            for kind in ['WITHIN_SCENE','ACROSS_SCENES','UNKNOWN_ANCHOR_SCENE']:
                subset=[p for p in pairs if p['kind']==kind];question[kind.lower()]={'correct_pairs':sum(p['correct'] for p in subset),'planned_pairs':len(subset),
                   'accuracy_fixed_denominator':ratio(sum(p['correct'] for p in subset),len(subset))}
        questions.append(question)
    require(not errors,'RAW_SCORE_CONSISTENCY_ERRORS:'+json.dumps(errors))
    status=scores['status'];cost=scores.get('cost',{})
    return {'status':status,'branch':manifest['branch'],'questions':questions,'requests':observations,'citations':citations_out,'bindings':bindings,
      'score_recomputation_errors':errors,'cost':cost,'summary':summarize(questions,observations),'by_type':{typ:summarize([q for q in questions if q['question_type']==typ],[r for r in observations if r['question_type']==typ]) for typ in ['multi_select','ordering']},'gold_strata':stratify(questions,observations)}

def summarize(questions,requests):
    roots=collections.defaultdict(list)
    for q in questions:roots[q['root_id']].append(q['exact_rate'])
    summary={'question_count':len(questions),'root_count':len(roots),'planned_calls':len(requests),'valid_calls':sum(r['valid'] for r in requests),'exact_calls':sum(r['exact'] for r in requests),
      'call_equal_exact':ratio(sum(r['exact'] for r in requests),len(requests)), 'question_equal_exact':mean([q['exact_rate'] for q in questions]),
      'root_equal_exact':mean([mean(v) for v in roots.values()]),'invalid_calls_count_as_incorrect':True}
    mc=[r for r in requests if r['question_type']=='multi_select' and r['valid']]
    ordering=[r for r in requests if r['question_type']=='ordering']
    if mc:summary['multi_select_valid_only']={'denominator_valid_calls':len(mc),'tp':sum(r['tp'] for r in mc),'fp':sum(r['fp'] for r in mc),'fn':sum(r['fn'] for r in mc),
       'jaccard':mean([r['jaccard'] for r in mc]),'overselection_rate':mean([r['overselected'] for r in mc]),'selected_all_rate':mean([r['selected_all'] for r in mc])}
    if ordering:
        pairs=[p for r in ordering for p in r['ordering_pairs']];summary['ordering']={'correct_pairs':sum(p['correct'] for p in pairs),'planned_pairs':len(pairs),'pair_accuracy_fixed_denominator':ratio(sum(p['correct'] for p in pairs),len(pairs)),
         'pair_accuracy_valid_only':mean([r['pair_accuracy_valid_only'] for r in ordering if r['valid']])}
        for kind in ['WITHIN_SCENE','ACROSS_SCENES','UNKNOWN_ANCHOR_SCENE']:
            subset=[p for p in pairs if p['kind']==kind];summary['ordering'][kind.lower()]={'correct_pairs':sum(p['correct'] for p in subset),'planned_pairs':len(subset),'accuracy_fixed_denominator':ratio(sum(p['correct'] for p in subset),len(subset))}
    return summary

def stratify(qs,requests):
    output=[]
    for typ in ['multi_select','ordering']:
        for g in sorted({q['gold_count'] for q in qs if q['question_type']==typ}):
            sub=[q for q in qs if q['question_type']==typ and q['gold_count']==g];ids={q['candidate_id'] for q in sub}
            output.append({'question_type':typ,'gold_count':g,**summarize(sub,[r for r in requests if r['candidate_id'] in ids])})
    return output

def citation_summary(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['canonical_label'] is not None:groups[(r['request_id'],r['display_label'])].append(r)
    return {'entries':sum(r['record_kind']=='MODEL_ENTRY' for r in rows),'expected_labels':len(groups),
       'labels_missing_entry':sum(all(r['record_kind']=='MISSING_MODEL_ENTRY' for r in group) for group in groups.values()),
       'labels_with_duplicate_entries':sum(sum(r['record_kind']=='MODEL_ENTRY' for r in group)>1 for group in groups.values()),
       'labels_with_any_valid_mechanical_citation':sum(any(r['citation_mechanically_valid'] for r in group) for group in groups.values()),
       'rows_including_missing_placeholders':len(rows),'turn_exists':sum(r['turn_exists'] for r in rows),'exact_nonempty_quote':sum(r['quote_exact_in_cited_turn'] for r in rows),
       'mechanically_valid_including_allowed_empty':sum(r['citation_mechanically_valid'] for r in rows),
       'empty_citation':sum(r['empty_citation'] for r in rows),'model_status_agrees_with_author':sum(r.get('model_status_agrees_with_author') is True for r in rows),
       'model_status_comparable':sum(r.get('model_status_agrees_with_author') is not None for r in rows),
       'author_location_relations':dict(collections.Counter(r['location_relation_to_author'] for r in rows)),
       'semantic_interpretation':'Location and exact-string checks do not establish whether a proposition is supported. Different valid turns may be alternative correct evidence. Invalid quotes are provenance errors, not an automatically established reasoning mechanism.'}

def percent(v):return '—' if v is None else f'{100*v:.2f}%'
def csv_write(p,rows,fields):
    with pathlib.Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def report_zh(obs):
    lines=['# 第二天 Solver 观察记录','',f"状态：{obs['status']}。本文件只记录可复核现象，不修改 Gold，不生成 cause。",'']
    if not obs.get('runs'):
        lines.extend(['评估结果尚未齐备，等待独立的 SCORES.json 与完整原始响应封存。没有推测准确率或模型弱点。',''])
        return '\n'.join(lines)
    lines.extend(['| 支路 | 题目 | roots | 有效/计划调用 | 题等权 exact | root 等权 exact |','|---|---:|---:|---:|---:|---:|'])
    for branch,r in obs['runs'].items():
        s=r['summary'];lines.append(f"| {branch} | {s['question_count']} | {s['root_count']} | {s['valid_calls']}/{s['planned_calls']} | {percent(s['question_equal_exact'])} | {percent(s['root_equal_exact'])} |")
    lines.extend(['','无效、缺失与未发送请求均保留在 exact 固定分母。多选 TP/FP/FN、Jaccard 与过选只在有效答案上统计，报告其有效分母；排序 pair 同时给固定分母和有效答案版本。','',
      '| 支路 | 题型 | Gold 数 | 题目 | 题等权 exact | root 等权 exact |','|---|---|---:|---:|---:|---:|'])
    for branch,r in obs['runs'].items():
        for g in r['gold_strata']:lines.append(f"| {branch} | {g['question_type']} | {g['gold_count']} | {g['question_count']} | {percent(g['question_equal_exact'])} | {percent(g['root_equal_exact'])} |")
    lines.extend(['','同一 mapping 的多选 3 次采样变化与跨 mapping 的经验答案分布差异分别保存。跨 mapping 差异同时含独立 seed 抽样噪声，不能据此直接断言位置敏感。排序每 mapping 只有 1 次，无法估计同一 mapping 内的抽样波动。',''])
    if 'evidence' in obs['runs']:
        c=obs['runs']['evidence']['citation_summary'];lines.extend([f"结构化依据共应覆盖 {c['expected_labels']} 个 label，实际 {c['entries']} 条引用条目；{c['labels_missing_entry']} 个 label 缺失，{c['labels_with_duplicate_entries']} 个 label 重复。{c['turn_exists']} 条 turn_ref 存在，{c['exact_nonempty_quote']} 条非空 quote 逐字出现在指定 turn 中。",'',
          '逐 label 对齐使用公开显示映射还原 canonical 内容，再与该内容的 author evidence 比较。同 turn、同 scene、同 speaker 与其他位置分开记录。schema 未要求模型提供 speaker，因此 speaker 来自源 turn 的确定信息，不把未提供 speaker 当作错误。',
          '', '引用位置不同、引文长度不同或字符串不同不能直接判作语义错误；逐字匹配也不能证明引用支持所选命题。语义和因果解释需要后续受控核查。',''])
    else:lines.extend(['结构化依据结果尚未完成，引用核对等待 evidence 的完整原始记录。',''])
    lines.extend(['该数据属于 train.feedback；与主评测、全新 roots 的泛化结果分开解释。所有问题与 Gold 原样保留。',''])
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--direct',type=pathlib.Path,default=DEFAULT_DIRECT);p.add_argument('--evidence',type=pathlib.Path,default=DEFAULT_EVIDENCE)
    p.add_argument('--out',type=pathlib.Path,required=True);arg=p.parse_args();require(not arg.out.exists(),'OUTPUT_EXISTS_NO_OVERWRITE')
    runs={};waiting=[]
    for branch,run in [('direct',arg.direct),('evidence',arg.evidence)]:
        if (run/'SCORES.json').exists() and (run/'RAW_RESPONSE_FREEZE.json').exists():
            r=load_run(run);require(r['branch']==branch,'BRANCH_MISMATCH');r['citation_summary']=citation_summary(r['citations']);runs[branch]=r
        else:waiting.append({'branch':branch,'run_path':str(run),'awaiting':['SCORES.json','RAW_RESPONSE_FREEZE.json']})
    status='COMPLETE_OBSERVATIONS' if len(runs)==2 else 'PARTIAL_AWAITING_EVIDENCE' if 'direct' in runs else 'WAITING_FOR_SOLVER_RESULTS'
    out={'schema_version':'day2_source_bound_observations_v1','created_at_utc':now(),'status':status,'namespace':'train.feedback','runs':runs,'waiting':waiting,
      'limits':['No cause diagnosis or Gold modifications.','Same-root questions and repeated sampling are dependent.','Between-mapping variation is descriptive and is confounded with independent sampling seeds.','Exact quotation and author-location agreement do not establish semantic correctness.'],
      'tool_binding':bind(pathlib.Path(__file__)),'new_solver_calls':0,'gpu_inference_started':False}
    arg.out.mkdir(parents=True);save(arg.out/'OBSERVATIONS.json',out)
    qrows=[];crow=[]
    for branch,r in runs.items():
        for q in r['questions']:
            flat={**q,'branch':branch};flat.update({k:v for k,v in q['permutation_observations'].items() if k not in ['within_mapping_details','between_mapping_details']});qrows.append(flat)
        crow.extend({**r,'branch':branch} for r in r['citations'])
    qfields=['branch','candidate_id','root_id','question_type','gold_count','planned_calls','valid_calls','exact_calls','exact_rate','stratum','tp','fp','fn','mean_jaccard_valid_only','overselection_rate_valid_only','all_selection_rate_valid_only','mean_selected_count_valid_only',
       'ordering_pair_correct','ordering_pair_denominator','pair_accuracy_fixed_denominator','pair_accuracy_valid_only','within_scene','across_scenes','unknown_anchor_scene',
       'within_mapping_sample_pairs','within_mapping_different_pairs','within_mapping_pair_disagreement_valid_only','within_mapping_variation_is_estimable','between_mapping_distribution_tv_mean_valid_only','pooled_distinct_canonical_answers','pooled_modal_fraction_valid_only','ordering_single_sample_limitation']
    csv_write(arg.out/'QUESTION_OBSERVATIONS.csv',qrows,qfields)
    cfields=['branch','record_kind','request_id','ordinal','candidate_id','root_id','question_type','display_label','canonical_label','model_status','turn_ref','quote','turn_exists','quote_exact_in_cited_turn','quote_offsets','quote_occurrence_turn_refs_in_complete_source','source_speaker_at_citation','source_scene_at_citation','author_primary_ref','location_relation_to_author','same_speaker_as_any_author_citation','model_speaker_claim_status','model_status_agrees_with_author','citation_mechanically_valid','empty_citation_status_compatible','semantic_judgment']
    csv_write(arg.out/'CITATION_OBSERVATIONS.csv',crow,cfields)
    with (arg.out/'REPORT_ZH.md').open('x') as f:f.write(report_zh(out))
    save(arg.out/'MANIFEST.json',{'created_at_utc':now(),'files':[bind(p) for p in sorted(arg.out.iterdir()) if p.is_file()],'tool_binding':bind(pathlib.Path(__file__))})
    print(json.dumps({'status':status,'runs':list(runs),'out':str(arg.out)},ensure_ascii=False))
if __name__=='__main__':main()
