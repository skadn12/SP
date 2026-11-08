"""Freeze routed Writer candidates and a source-reviewed ordering correction.

No model/API calls. Original Writer artifacts remain immutable. Choice rows
are unchanged; ordering review is Codex offline diagnosis, not human gold.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
W = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1')
sys.path.insert(0, str(W))
from scripts.current_method.restart_n5_spell.h4_ordering_grounded_options_v2.contract import derive_ordering

D = Path(__file__).resolve().parent
P = D / 'prepared'
SOURCE = D.parent / 'h4_writer_coverage_v1/prepared_run/live/roots'
ROOTS = ['cc-train-episode-259192', 'cc-train-episode-96462', 'cc-train-episode-112812']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    assert not P.exists(), 'prepared already exists; do not overwrite'
    P.mkdir()
    sources, rows, original, reviews, counts = [], [], [], [], {}
    notes = {
        ('259192', 'I001'): ('PASS', '跳崖经历→回顾反思→更安全的冒险需求→David建议攀岩。过去行动和后续表达在原文可区分。',
                            ['cc-train-episode-259192:P02:T003', 'cc-train-episode-259192:P03:T009', 'R01:S02:T001', 'R01:S02:T004']),
        ('259192', 'I002'): ('PASS', '上周末参加语言交流→认识Sarah→交换联系方式→当前向David报告。过去事件与当前报告共用T001引用是合法的，不按提及顺序重排过去事件。',
                            ['R02:S03:T001', 'R02:S03:T003', 'R02:S03:T006', 'R02:S03:T007']),
        ('96462', 'I001'): ('PASS', '家庭和数学困扰→3月16日买咖啡→3月30日所述昨天的游行→3月30日当前决定参加自卫课。',
                           ['cc-train-episode-96462:P01:T001', 'cc-train-episode-96462:P02:T001', 'cc-train-episode-96462:P02:T003', 'cc-train-episode-96462:P04:T001', 'cc-train-episode-96462:P04:T003', 'R01:S01:T004', 'R01:S01:T013', 'R01:S01:T014']),
        ('112812', 'I001'): ('PASS', '贷款负担→预算建议→回顾该建议且找到记账应用→商定周日晚定期检查。R02日期UNKNOWN，但明确承接关系支持顺序。',
                            ['cc-train-episode-112812:P04:T001', 'cc-train-episode-112812:P04:T002', 'R02:S03:T001', 'R02:S03:T004', 'R02:S03:T005']),
        ('112812', 'I002'): ('CORRECTED_GOLD', 'T004首次建议问服务员；T005回应并建议经典与新品混点。T008重申问服务员，不应把首次建议移到T005之后。题干和选项保持原样，修正节点I2绑定和方向关系后由程序重算Gold。',
                            ['R01:S03:T001', 'R01:S03:T004', 'R01:S03:T005', 'R01:S03:T008', 'R01:S03:T012']),
    }
    for root in ROOTS:
        base = SOURCE / root
        routed_path, source_path = base / 'routed_candidates.json', base / 'qa_source.json'
        routed, expanded = read(routed_path), read(source_path)
        for path in (routed_path, source_path, base / 'candidates_valid.json', base / 'routed_choices_program_accepted.json'):
            sources.append({'path': str(path), 'sha256': sha(path)})
        target = P / 'dialogues' / (root + '.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source_path, target)
        assert sha(target) == sha(source_path)
        catalog = {t['turn_ref']: t for t in expanded['turns']}
        counts[root] = {'program_accepted': dict(Counter(c['question_type'] for c in read(base / 'candidates_valid.json'))),
                        'selected': dict(Counter(c['question_type'] for c in routed)),
                        'dialogue_turns': len(expanded['turns']), 'source_sha256': sha(source_path)}
        for candidate in routed:
            original.append(deepcopy(candidate))
            current = deepcopy(candidate)
            if candidate['question_type'] == 'ordering':
                path = base / candidate['ordering_derivation']['path']
                sources.append({'path': str(path), 'sha256': sha(path)})
                derivation = read(path)
                assert sha(path) == candidate['ordering_derivation']['sha256']
                status, rationale, refs = notes[(root.rsplit('-', 1)[1], candidate['intent_id'])]
                review = {'candidate_id': candidate['candidate_id'], 'root_id': root, 'status': status,
                          'reviewer': 'Codex offline source diagnosis, independently cross-checked by another Codex agent',
                          'not_independent_human_certification': True,
                          'original_gold': candidate['gold'], 'evaluation_gold': candidate['gold'],
                          'rationale': rationale, 'source_evidence': [catalog[r] for r in refs],
                          'original_derivation': {'path': str(path), 'sha256': sha(path)}}
                if status == 'CORRECTED_GOLD':
                    row = deepcopy(derivation['raw_row'])
                    nodes = row['event_nodes']
                    node = next(n for n in nodes if n['item_id'] == 'I2')
                    node['evidence_refs'] = ['R01:S03:T004']
                    node['evidence_spans'] = [{'turn_ref': 'R01:S03:T004', 'quote': catalog['R01:S03:T004']['text']}]
                    row['temporal_relations'] = [
                        {'ordered_pair': [left['item_id'], right['item_id']],
                         'evidence_refs': left['evidence_refs'] + right['evidence_refs'],
                         'evidence_spans': left['evidence_spans'] + right['evidence_spans']}
                        for left, right in zip(nodes, nodes[1:])]
                    row['evidence_groups'] = {'events': [r for n in nodes for r in n['evidence_refs']]}
                    identity = {**derivation['identity'], 'run_id': D.name,
                                'stage': 'CODEX_OFFLINE_CORRECTION', 'original_derivation_sha256': sha(path)}
                    result = derive_ordering(row, expanded=expanded,
                        policy={'version': 'ordering_grounded_options_v2', 'node_count': 4}, identity=identity)
                    assert result['status'] == 'PASS', result['errors']
                    assert result['projected_row']['gold'] == ['I1', 'I2', 'I3', 'I4']
                    assert result['projected_row']['items'] == candidate['items']
                    corrected_path = P / 'corrections' / '112812-I002-ordering-derivation.json'
                    write(corrected_path, result['record'])
                    for key in ('gold', 'evidence_refs', 'evidence_groups'):
                        current[key] = result['projected_row'][key]
                    current['ordering_derivation'] = {'path': str(corrected_path), 'sha256': sha(corrected_path), 'identity': identity}
                    current['codex_offline_correction'] = {'original_gold': candidate['gold'], 'original_derivation': review['original_derivation'], 'reason': rationale}
                    review['evaluation_gold'] = current['gold']
                    review['corrected_derivation'] = {'path': str(corrected_path), 'sha256': sha(corrected_path)}
                    review['changed_fields'] = [k for k in current if current.get(k) != candidate.get(k)]
                reviews.append(review)
            else:
                assert current == candidate
            rows.append(current)
    assert len(rows) == 28
    assert Counter(r['question_type'] for r in rows) == {'single_choice': 16, 'multi_select': 7, 'ordering': 5}
    assert len(reviews) == 5
    write(P / 'original_routed_candidates.json', original)
    write(P / 'candidate_manifest.json', rows)
    write(P / 'ORDERING_REVIEW.json', reviews)
    sources.extend({'path': str(p), 'sha256': sha(p)} for p in [Path(__file__).resolve(), Path(sys.modules[derive_ordering.__module__].__file__)])
    review = {'status': 'PASS', 'candidate_count': len(rows), 'by_root': counts,
        'by_type': dict(Counter(r['question_type'] for r in rows)),
        'ordering_review': {'original_gold_supported': 4, 'gold_corrected_before_evaluation': 1, 'excluded': 0},
        'selection': 'All routed candidates; exact duplicate routing was already performed by Writer. No selection based on Solver output.',
        'choices_policy': 'Program acceptance only; no semantic or evidence completeness rejection; all 23 choice rows unchanged.',
        'dates_policy': 'Complete original dated inputs preserved; UNKNOWN dates remain UNKNOWN.',
        'ordering_policy': 'Codex offline source diagnosis; not independent human certification or a new production semantic gate.',
        'source_counts': {'program_accepted': 34, 'routed': 28, 'raw_ordering_including_repair': 11, 'accepted_ordering': 7, 'routed_ordering': 5},
        'prior_report_correction': 'Prior derived reports mixed raw, accepted and routed denominators and wrongly held the language-exchange report question. This frozen five-question review supersedes those eligibility judgments.',
        'external_api_calls': 0, 'solver_calls': 0, 'sources': sources}
    write(P / 'SELECTION_REVIEW.json', review)
    print(json.dumps({'status': 'PASS', 'candidate_count': len(rows), 'by_root': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
