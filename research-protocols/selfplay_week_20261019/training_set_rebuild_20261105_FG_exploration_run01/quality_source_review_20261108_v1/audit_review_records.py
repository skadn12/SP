import json, pathlib, collections, hashlib
R=pathlib.Path(__file__).resolve().parent
man=json.loads((R/'MANIFEST.json').read_text())
entries=man['entries']; bypid={e['packet_id']:i for i,e in enumerate(entries)}
rows=collections.defaultdict(list)
for fn in sorted(R.glob('reviewer_*/REVIEW*.jsonl')):
    for ln,line in enumerate(fn.read_text().splitlines(),1):
        if not line.strip(): continue
        x=json.loads(line); rows[x.get('packet_id')].append({'file':str(fn.relative_to(R)),'line':ln,'record':x})
resolved=[]; missing=[]; duplicates=[]; malformed=[]; evidence_gaps=[]
for i,e in enumerate(entries):
    pid=e['packet_id']; rs=rows.get(pid,[])
    if not rs: missing.append(i); continue
    if len(rs)>1: duplicates.append({'index':i,'packet_id':pid,'records':[(r['file'],r['line'],r['record'].get('status')) for r in rs]})
    statuses={r['record'].get('status') for r in rs}
    # conservative resolution: HOLD dominates any PASS; disagreement among non-HOLD is retained.
    if 'HOLD' in statuses: status='HOLD'; decision_reason='HOLD_dominates_conflict'
    elif len(statuses)==1: status=next(iter(statuses)); decision_reason='unanimous'
    else: status='UNKNOWN'; decision_reason='unresolved_status_conflict'
    chosen=next((r for r in rs if r['record'].get('status')==status),rs[0])
    x=chosen['record']; gaps=[]
    if not isinstance(x.get('option_evidence'),list) or not x['option_evidence']: gaps.append('missing_option_evidence')
    else:
      for oe in x['option_evidence']:
        if not oe.get('turn_refs') or not oe.get('quotes'): gaps.append(oe.get('option_id','<missing_option_id>'))
    if gaps: evidence_gaps.append({'index':i,'packet_id':pid,'status':status,'gaps':gaps,'file':chosen['file'],'line':chosen['line']})
    resolved.append({'index':i,'packet_id':pid,'root_id':e['root_id'],'type':e['type'],'status':status,'decision_reason':decision_reason,'source_file':chosen['file'],'source_line':chosen['line'],'all_statuses':sorted(statuses),'evidence_complete':not gaps})
summary={'schema':'quality_review_audit_v1','manifest_sha256':hashlib.sha256((R/'MANIFEST.json').read_bytes()).hexdigest(),'denominator':len(entries),'record_lines':sum(len(v) for v in rows.values()),'unique_reviewed':len(rows),'missing_count':len(missing),'missing_indices':missing,'duplicate_packet_count':sum(1 for d in duplicates),'duplicates':duplicates,'evidence_gap_count':len(evidence_gaps),'evidence_gaps':evidence_gaps,'resolved_status_counts':dict(collections.Counter(x['status'] for x in resolved)),'resolved':resolved}
out=R/'REVIEW_AUDIT_PREP.json'; out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:summary[k] for k in ['denominator','record_lines','unique_reviewed','missing_count','duplicate_packet_count','evidence_gap_count','resolved_status_counts']},ensure_ascii=False,indent=2))
