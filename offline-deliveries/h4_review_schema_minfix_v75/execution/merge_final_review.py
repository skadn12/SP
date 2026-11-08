"""Merge final reviews only after the shared strict validator accepts them."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate
import review_schema


def main(home=ROOT):
    home = Path(home).resolve()
    evaluate.require(not (home / 'EVALUATION_PREPARATION.STARTED.json').exists(),
                     'final_review_cannot_merge_after_preparation')
    evaluate.require(not (home / 'FINAL_CONTENT_REVIEW.COMPLETE.json').exists(),
                     'final_review_cannot_merge_after_freeze')
    evaluate.no_started_solver(home)
    validated = review_schema.validate_final_reviews(home)
    sources = [evaluate.read(row['path']) for row in validated['review_records']
               if Path(row['path']).name.endswith('_SOURCE_JUDGMENTS.json')]
    evaluate.require(bool(sources), 'final_review_sources_missing')
    disclosures = {}
    for key in ('parent_all_true_provenance_known', 'parent_gold_known_before_review',
                'fresh_blind_parent_review', 'current_solver_scores_read',
                'solver_scores_used', 'child_gold_read_before_save', 'current_pool_read_before_save'):
        values = [source[key] for source in sources]
        evaluate.require(all(value == values[0] for value in values), 'mixed_final_review_disclosure:' + key)
        disclosures[key] = values[0]
    result = {'pool_sha256': evaluate.sha(home / 'POOL.json'), 'reviewed_at_utc': evaluate.now(),
              **disclosures, 'review_schema_version': validated['review_schema_version'],
              'derived_from_saved_final_reviews': True,
              'independent_review_files': [row for row in validated['review_records']
                                           if Path(row['path']).parent == home / 'final_review'],
              'questions': validated['questions']}
    evaluate.write(home / 'FINAL_CONTENT_REVIEW.json', result)
    print(json.dumps({'questions': len(result['questions']),
                      'usable': sum(row['core_status'] == 'CORE_USABLE' for row in result['questions'])}))
    return result


if __name__ == '__main__':
    main()
