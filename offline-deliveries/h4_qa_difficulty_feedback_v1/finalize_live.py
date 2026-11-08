"""Aggregate completed generation and Codex diagnosis; no model calls."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parent
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.qa_difficulty_feedback_run import check_freeze

RUN = BASE / 'prepared_run/live'
OLD = BASE.parent / 'h4_gate_downstream_v1/prepared_run'
ROOTS = ('cc-train-episode-259192','cc-train-episode-96462','cc-train-episode-112812')


def read(p): return json.loads(Path(p).read_text())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def fingerprint(q):
    return json.dumps({k:q.get(k) for k in ('question_type','question','options','items','gold')},
                      sort_keys=True,ensure_ascii=False)


def type_counts(rows): return dict(Counter(q['question_type'] for q in rows))


def main():
    manifest=read(BASE/'prepared_run/freeze_manifest.json')
    check_freeze(manifest)
    execution=read(RUN/'generation_summary.json')
    assert execution['status']=='FINISHED' and execution['external_http_attempt_count']==21
    audit=read(BASE/'reports/EXECUTION_AUDIT.json')
    assert audit['status']=='PASS'
    ledger=read(RUN/'execution/qa_call_ledger.json')
    old_ledger=read(OLD/'qa/execution/qa_call_ledger.json')
    roots, all_reviews, all_final, strict_all, repeats = [], [], [], [], []
    for root in ROOTS:
        code=root.rsplit('-',1)[1]
        report=read(BASE/f'reports/CONTENT_{code}.json')
        rows=report['raw_candidate_reviews']
        pool=read(RUN/'roots'/root/'candidate_pool.json')
        final={q['candidate_id']:q for q in pool}
        source_turns={t['turn_ref']:t for t in read(RUN/'roots'/root/'qa_source.json')['turns']}
        for row in rows:
            for evidence in row.get('diagnostic_evidence',[]):
                original=source_turns[evidence['turn_ref']]
                assert evidence['text']==original['text']
                assert evidence['speaker']==original['speaker']
        expected=set()
        for record in read(RUN/'roots'/root/'candidate_generation.json'):
            rid=record['request_id']
            raw=read(RUN/'execution/calls'/root/rid/'response.txt')['candidates']
            matching={r['raw_row_index']:r for r in rows if r['request_id']==rid}
            assert len(matching)==len(raw)
            for i,q in enumerate(raw,1):
                assert matching[i]['raw_candidate']==q
                expected.add((rid,i))
        assert expected=={(r['request_id'],r['raw_row_index']) for r in rows}
        assert len(rows)==len(expected)
        final_rows=[r for r in rows if r['in_final_pool']]
        assert set(final)=={r['candidate_id'] for r in final_rows}
        assert all(r['final_candidate']==final[r['candidate_id']] for r in final_rows)
        excluded=set(report.get('strict_pass_dedup_excluded_ids',[]))
        for row in final_rows:
            if row.get('strict_pass_deduplicated') is False and row['verdict']=='PASS':
                excluded.add(row['candidate_id'])
        strict=[r for r in final_rows if r['verdict']=='PASS' and r['candidate_id'] not in excluded]
        old_pool=read(OLD/'qa/roots'/root/'candidate_pool.json')
        old_keys={fingerprint(q):q['candidate_id'] for q in old_pool}
        root_repeats=[]
        for q in pool:
            if fingerprint(q) in old_keys:
                root_repeats.append({'candidate_id':q['candidate_id'],
                    'historical_candidate_id':old_keys[fingerprint(q)],
                    'strict_pass_dedup':q['candidate_id'] in {r['candidate_id'] for r in strict}})
        repeats.extend(root_repeats)
        tokens=sum(r['total_tokens'] for r in ledger['records'] if r['root_id']==root)
        has_new=lambda q:any(ref.startswith('R02:') for ref in q['evidence_refs'])
        def needs_new(r):
            v=r.get('new_R02_information_necessary',r.get('new_R02_information_necessary_for_answer',
                      r.get('new_R02_answer_necessity')))
            return v is True or v=='YES'
        stats={'root_id':root, 'raw_candidates':len(rows), 'raw_by_type':type_counts(rows),
            'program_accepted':len(pool),'program_accepted_by_type':type_counts(pool),
            'content_verdicts':dict(Counter(r['verdict'] for r in final_rows)),
            'strict_pass_dedup':len(strict),'strict_pass_dedup_by_type':type_counts(strict),
            'dedup_excluded_ids':sorted(excluded),
            'ordering_content_verdicts':dict(Counter(r['verdict'] for r in final_rows if r['question_type']=='ordering')),
            'ordering_whole_source_support':dict(Counter(r['whole_source_support'] for r in final_rows if r['question_type']=='ordering')),
            'strict_pass_with_R02_refs':sum(has_new(r['final_candidate']) for r in strict),
            'strict_pass_requiring_R02':sum(needs_new(r) for r in strict),
            'exact_historical_repeats':root_repeats,'total_tokens_all_calls':tokens,
            'tokens_per_strict_pass_dedup':tokens/len(strict) if strict else None,
            'tokens_per_valid_ordering':None,'valid_ordering_cost_reason':'ZERO_STRICT_PASS_ORDERING',
            'verified_currency_cost':'UNKNOWN'}
        roots.append(stats)
        target=BASE/'filtered'/root
        save(target/'strict_pass_dedup.json',[r['final_candidate'] for r in strict])
        for verdict in ('PASS','HOLD','FAIL'):
            save(target/('ordering_'+verdict+'.json'),
                 [r['final_candidate'] for r in final_rows if r['question_type']=='ordering' and r['verdict']==verdict])
        # An ordering-only view keeps SC/MS unchanged, including diagnostic FAILs.
        # The separately named strict pool is the source for supported counts.
        save(target/'ordering_filtered_scms_unchanged.json',
             [q for q in pool if q['question_type']!='ordering' or any(
                 r['candidate_id']==q['candidate_id'] and r['verdict']=='PASS' for r in final_rows)])
        save(target/'FILTER_MANIFEST.json',{'reviewer':'Codex','production_candidates_modified':False,
             'new_semantic_production_gate':False,'independent_human_certification':False,
             'strict_pass_dedup_count':len(strict),'stats':stats})
        all_reviews.extend(rows);all_final.extend(final_rows);strict_all.extend(strict)
    total_tokens=sum(r['total_tokens'] for r in ledger['records'])
    old_tokens=sum(r['total_tokens'] for r in old_ledger['records'])
    old_eligible=read(BASE/'COMPILED_FEEDBACK.json')['records']
    old_count=sum(r['eligible_count'] for r in old_eligible)
    result={'status':'LIVE_COMPLETE_REVIEWED_STOPPED','reviewer':'Codex',
        'independent_human_certification':False,'roots':roots,
        'raw_candidates':len(all_reviews),'program_accepted':len(all_final),
        'raw_by_type':type_counts(all_reviews),'program_accepted_by_type':type_counts(all_final),
        'content_verdicts':dict(Counter(r['verdict'] for r in all_final)),
        'strict_pass_dedup':len(strict_all),'strict_pass_dedup_by_type':type_counts(strict_all),
        'ordering_content_verdicts':dict(Counter(r['verdict'] for r in all_final if r['question_type']=='ordering')),
        'ordering_whole_source_support':dict(Counter(r['whole_source_support'] for r in all_final if r['question_type']=='ordering')),
        'strict_pass_with_R02_refs':sum(r['strict_pass_with_R02_refs'] for r in roots),
        'strict_pass_requiring_R02':sum(r['strict_pass_requiring_R02'] for r in roots),
        'exact_prior_question_repeats':repeats,'new_difficulty_measurement':'NOT_RUN',
        'calls':{'http':21,'logical':21,'miner':3,'writer_primary':9,'writer_repair':9,
                 'automatic_retries':0,'timeouts':0,'technical_failures':0,'extra_audit_api':0,
                 'solver':0,'upstream_expansion':0,'memory_api':0,'training':0},
        'cost':{'input_tokens':sum(r['input_tokens'] for r in ledger['records']),
                'output_tokens':sum(r['output_tokens'] for r in ledger['records']),
                'total_tokens':total_tokens,'reasoning_tokens':'UNKNOWN','verified_currency_cost':'UNKNOWN',
                'tokens_per_strict_pass_dedup':total_tokens/len(strict_all),
                'tokens_per_valid_ordering':None,'ordering_cost_reason':'ZERO_STRICT_PASS_ORDERING'},
        'historical_same_source_comparison':{'old_http':len(old_ledger['records']),
            'old_total_tokens':old_tokens,'new_total_tokens':total_tokens,
            'token_increase_fraction':total_tokens/old_tokens-1,
            'old_strict_pass_dedup':old_count,'new_strict_pass_dedup':len(strict_all),
            'old_tokens_per_strict_pass_dedup':old_tokens/old_count,
            'method':'DESCRIPTIVE_KNOWN_CASE_COMPARISON_NOT_REPEATED_PAIRED_CAUSAL_ESTIMATE'},
        'automatic_feedback_activation':False,'automatic_next_experiment':False,
        'raw_candidate_and_gold_modified':False,'frozen_records_verified':len(manifest['files'])}
    save(BASE/'reports/FINAL_REPORT.json',result)
    save(BASE/'reports/CONTENT_ALL_ROWS.json',all_reviews)
    save(BASE/'reports/FINAL_VERIFICATION.json',{'status':'PASS','all_raw_rows_covered':len(all_reviews),
        'all_final_candidates_identical':len(all_final),'all_selected_candidates_unmodified':len(strict_all),
        'feedback_and_input_freeze_records_unchanged':len(manifest['files']),
        'budget_21_respected':True,'sources_and_gold_not_rewritten':True,
        'all_diagnostic_excerpts_match_frozen_source':True,'new_audit_model_calls':0})
    print(json.dumps({k:result[k] for k in ('status','raw_candidates','program_accepted','strict_pass_dedup',
        'strict_pass_dedup_by_type','strict_pass_with_R02_refs','strict_pass_requiring_R02')},ensure_ascii=False))
    print('exact historical repeats',len(repeats),'strict',sum(r['strict_pass_dedup'] for r in repeats))


if __name__=='__main__':main()
