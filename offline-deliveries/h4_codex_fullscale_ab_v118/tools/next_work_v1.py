"""Read-only assigned-role queue; never open question-side private designs."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from status import population


def queue(executor):
    batch = HERE.parent
    roles = {r['root_id']: r for r in json.loads((batch / 'ROLE_ASSIGNMENTS.json').read_text())['roots']}
    jobs = []
    for state in population()['roots']:
        if state['terminal']:
            continue
        root = state['root_id']
        role = roles[root]
        unit = batch / 'units' / ('root' + root.rsplit('-', 1)[1])
        folder = unit / 'roots' / unit.name
        author = role['author'] == executor
        reviewer = role['reviewer'] == executor
        if state['status'] == 'MATCHING_PENDING' and author:
            jobs.append({'priority': 50, 'root_id': root, 'role': 'author', 'stage': 'matching',
                         'input_render_if_already_started': str(unit / 'evidence/MATCHING/FULL_INPUT.txt')})
        elif state['status'] == 'EXPANSION_ACTIVE':
            for scene in state['scenes']:
                if reviewer and scene['status'] == 'AWAITING_REVIEW':
                    jobs.append({'priority': 10, 'root_id': root, 'role': 'reviewer', 'stage': 'scene-review',
                                 'scene_ref': scene['scene_ref'], 'tool': 'review_v3.py scene-read/scene-save'})
                elif author and scene['status'] == 'NOT_STARTED':
                    label = scene['scene_ref'].replace(':', '_')
                    jobs.append({'priority': 40, 'root_id': root, 'role': 'author', 'stage': 'scene-author',
                                 'scene_ref': scene['scene_ref'], 'frozen_selection': str(unit / 'selection.json'),
                                 'input_render_if_already_started': str(unit / 'evidence' / ('AUTHOR_' + label + '_FULL_READ.md'))})
        elif state['status'] == 'QUESTION_STAGES_PENDING':
            if author and not (folder / 'QA_DRAFT.json').exists():
                jobs.append({'priority': 30, 'root_id': root, 'role': 'author', 'stage': 'qa-author',
                             'tool': 'author.py qa-read, freeze anchors, preflight, qa-save'})
            elif reviewer and (folder / 'PUBLIC_QUESTIONS.json').exists() and not (folder / 'PUBLIC_SOURCE_JUDGMENTS.json').exists():
                jobs.append({'priority': 15, 'root_id': root, 'role': 'reviewer', 'stage': 'public-review',
                             'tool': 'review_v3.py public-read/public-save',
                             'source_first': 'Read only complete review render; do not open QA_DRAFT, anchors or private design/Gold.'})
            elif reviewer and (folder / 'PUBLIC_ASSETS.json').exists() and not (folder / 'ASSET_SOURCE_JUDGMENTS.json').exists():
                jobs.append({'priority': 20, 'root_id': root, 'role': 'reviewer', 'stage': 'asset-review',
                             'tool': 'review_v3.py asset-read/asset-save',
                             'source_first': 'Read only complete review render; do not open QA_DRAFT, anchors or private design/Gold.'})
            else:
                for prereq, output, action in [('QA_DRAFT.json', 'PUBLIC_ASSETS.json', 'publish-assets'),
                                               ('ASSET_SOURCE_JUDGMENTS.json', 'PUBLIC_QUESTIONS.json', 'compose'),
                                               ('PUBLIC_SOURCE_JUDGMENTS.json', 'FINAL_CROSSCHECK.json', 'finalize')]:
                    if (author or reviewer) and (folder / prereq).exists() and not (folder / output).exists():
                        jobs.append({'priority': 5, 'root_id': root, 'role': 'mechanical-controller',
                                     'stage': action, 'unit_path': str(unit)})
    return sorted(jobs, key=lambda j: (j['priority'], int(j['root_id'].rsplit('-', 1)[1])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor', required=True)
    args = p.parse_args()
    print(json.dumps({'executor': args.executor, 'jobs': queue(args.executor)}, ensure_ascii=False, indent=2))
