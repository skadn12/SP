"""Reparse every sealed output without relying on the saved validity flag."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from jsonschema import validate as schema_validate

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'execution'


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def main():
    assert not (HERE / 'FINAL_MANIFEST.json').exists(), 'DO_NOT_ADD_AFTER_BATCH_SEAL'
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    scores = read(OUT / 'SCORES.json')['records']
    reasons = read(OUT / 'REASON_REVIEW_PACKET.json')['requests']
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    assert freeze['request_ids'] == [r['request_id'] for r in rows]
    bound = {b['path']: b for b in freeze['bindings']}
    failures, valid_count, raw_paths = [], 0, []
    for row, score, reason in zip(rows, scores, reasons, strict=True):
        rid = row['request_id']
        p = OUT / 'raw' / (rid + '.json')
        raw_paths.append(p)
        assert bind(p) == bound[str(p.resolve())] == score['raw_binding']
        raw = read(p)
        assert raw['request_id'] == score['request_id'] == reason['request_id'] == rid
        error, answer, parsed = None, [], None
        try:
            assert raw['status'] == 'RETURNED', 'BACKEND_OR_SUBMISSION_FAILURE:' + raw['status']
            assert raw['gold_read'] is False
            assert raw['sampling_seed'] == row['sampling_seed']
            results = raw['returned_results']
            assert raw['returned_result_count'] == len(results) == 1, 'RETURNED_RESPONSE_DRIFT'
            result = results[0]
            assert result['prompt'] == Path(row['prompt_path']).read_text(), 'RETURNED_RESPONSE_DRIFT'
            assert result['prompt_token_ids'] == tokens[rid] == raw['prompt_token_ids'], 'RETURNED_RESPONSE_DRIFT'
            assert len(result['outputs']) == 1, 'RETURNED_RESPONSE_DRIFT'
            completion = result['outputs'][0]
            assert all(raw[k] == completion[k] for k in ('raw_output', 'finish_reason', 'stop_reason', 'output_token_ids')), 'RETURNED_RESPONSE_DRIFT'
            assert raw['finish_reason'] == 'stop', 'TRUNCATED_OR_NONSTOP_COMPLETION'
            parsed = json.loads(raw['raw_output'])
            schema_validate(parsed, read(row['schema_path']))
            assert list(parsed) == ['item_evidence', 'answer_reason', 'final_answer'], 'REASONS_MUST_PRECEDE_ANSWER'
            assert parsed['answer_reason'].strip(), 'EMPTY_ANSWER_REASON'
            for item in parsed['item_evidence'].values():
                assert item['reason'].strip(), 'EMPTY_ITEM_REASON'
                if row['condition'] == 'FULL':
                    assert len(item['turn_refs']) == len(set(item['turn_refs'])), 'DUPLICATE_REFERENCE'
                    assert item['judgment'] != 'supported' or item['turn_refs'], 'SUPPORTED_WITHOUT_CITATION'
                else:
                    assert item['turn_refs'] == [], 'QUESTION_ONLY_REFERENCE_FORBIDDEN'
            match = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})</box>', parsed['final_answer'])
            assert match, 'ANSWER_BOX_REQUIRED'
            answer = match.group(1).split()
            assert len(answer) == len(set(answer)) and set(answer) <= set(row['labels']), 'INVALID_LABEL_SET'
            if row['question_type'] == 'ordering':
                assert set(answer) == set(row['labels']), 'ORDERING_INCOMPLETE'
                if row['condition'] == 'FULL':
                    assert all(x['turn_refs'] for x in parsed['item_evidence'].values()), 'NO_ANCHOR_FOR_ORDERING'
        except Exception as exc:
            error = str(exc)
        valid = error is None
        assert valid == score['valid'], rid
        assert error == score['invalid_error'], (rid, error, score['invalid_error'])
        assert score['display_prediction'] == (answer if valid else []), rid
        if valid:
            valid_count += 1
            assert reason['parsed'] == parsed and reason['answer_labels'] == answer
        else:
            failures.append({'request_id': rid, 'independently_reparsed_error': error})
    assert len(rows) == len(scores) == len(reasons) == len(freeze['bindings']) == 272
    result = {
        'schema_version': 'production_independent_raw_reparse_v2',
        'status': 'PASS_INDEPENDENT_RAW_VALIDATION_REPARSE',
        'reviewer_id': '/root/production_gate_finish',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_text_read_only_after_all_responses_sealed': True,
        'root_parser_or_saved_validity_used_to_determine_validity': False,
        'raw_responses_reparsed': len(rows), 'valid': valid_count,
        'invalid': len(failures), 'invalid_errors': dict(Counter(r['independently_reparsed_error'] for r in failures)),
        'invalid_records': failures,
        'all_saved_validity_predictions_and_error_labels_reproduced': True,
        'all_parsed_reason_packets_equal_raw_json': True,
        'semantic_reason_correctness_certified': False,
        'candidate_or_gold_modified': False,
        'bindings': [bind(Path(__file__))] + [bind(OUT / name) for name in
            ('PUBLIC_REQUESTS.json', 'PROMPT_TOKEN_IDS.json', 'RAW_RESPONSE_FREEZE.json', 'SCORES.json', 'REASON_REVIEW_PACKET.json')],
    }
    p = HERE / 'audit/INDEPENDENT_RAW_VALIDATION_AUDIT.json'
    with p.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'raw_responses_reparsed', 'valid', 'invalid', 'invalid_errors')}))


if __name__ == '__main__':
    main()
