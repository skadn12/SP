"""Copy this versioned engine into a fresh shard; never execute it centrally."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
FILES=('common.py','batch.py','ordering_generation.py','ordering_plan.py','ordering_review.py',
       'mixed_evaluation.py','actor_reference.py','source_admission.py','source_subset.py','fresh_parent.py',
       'prior_rendering.py','batch_scope.py','expansion_binding.py')


def bind(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def install(home):
    home=Path(home).resolve()
    if home==HERE or home==HERE.parent or HERE in home.parents:
        raise ValueError('install_requires_separate_shard_directory')
    if (home/'BATCH_PREPARATION.STARTED.json').exists() or (home/'SHARD_ENGINE_INSTALLATION.json').exists():
        raise ValueError('refuse_reinstall_or_prepared_shard')
    if any((home/name).exists() for name in FILES):
        raise ValueError('shard_already_contains_production_file')
    home.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        with (home/name).open('xb') as stream:stream.write((HERE/name).read_bytes())
    record={'version':'v96_shard_engine_installation_v1',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'shard_home':str(home),
            'engine_home':str(HERE),'training_allowed':False,
            'source_files':[bind(HERE/name) for name in FILES],
            'files':[bind(home/name) for name in FILES],
            'installer':bind(Path(__file__)),'upstream_v79_modified':False}
    with (home/'SHARD_ENGINE_INSTALLATION.json').open('x') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('home',type=Path);args=parser.parse_args()
    print(json.dumps(install(args.home),ensure_ascii=False))
