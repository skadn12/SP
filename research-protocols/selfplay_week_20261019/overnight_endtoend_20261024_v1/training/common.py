"""Shared, CPU-safe evidence and strict answer scoring for OET1 only."""
from __future__ import annotations
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
REPO = TASK.parents[2]
WEEK = TASK.parent
MODEL = REPO / 'restart-n5-runs/day5_B1_schedule_v1/merged_model'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
NATIVE = REPO / 'current_method_github/train_qwen3_lora_sft.py'
TARGETS = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
PROTOCOL = TASK / 'PROTOCOL_FREEZE_V1.json'


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(items):
    for item in items:
        require(binding(item['path']) == item, 'BOUND_FILE_CHANGED:' + item['path'])


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        with temp.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    else:
        with path.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())


def save_lines(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        f.flush(); os.fsync(f.fileno())


def append(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        f.flush(); os.fsync(f.fileno())


def native():
    name = 'oet1_preserved_native_trainer'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, NATIVE)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def adapter():
    path = WEEK / 'day2/solver/adapter.py'
    name = 'oet1_preserved_public_adapter'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def tokenize(tokenizer, messages, cap):
    """Tokenize full source without the native trainer's left truncation."""
    n = native()
    require([m['role'] for m in messages] == ['system', 'user', 'assistant'], 'MESSAGES_REQUIRE_SYSTEM_USER_ASSISTANT')
    require(re.fullmatch(r'<box>[^<>]+</box>', messages[-1]['content']) is not None, 'DIRECT_BOX_TARGET_REQUIRED')
    prompt = n.render_chat(tokenizer, messages[:-1], True)
    full = n.render_chat(tokenizer, messages, False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids.append(tokenizer.eos_token_id)
    require(full_ids[:len(prompt_ids)] == prompt_ids, 'CHAT_PROMPT_NOT_FULL_PREFIX')
    require(len(prompt_ids) < len(full_ids) <= cap, 'OVER_LENGTH_NO_TRUNCATION_ALLOWED')
    feature = n.tokenize_record(tokenizer, {'messages': messages}, cap)
    require(feature['input_ids'] == full_ids, 'NATIVE_TOKENIZATION_DRIFT')
    require(feature['labels'] == [-100] * len(prompt_ids) + full_ids[len(prompt_ids):], 'SUPERVISION_MASK_DRIFT')
    metrics = {'expected_prompt_tokens': len(prompt_ids), 'expected_full_input_tokens': len(full_ids),
               'expected_supervised_tokens': len(full_ids) - len(prompt_ids),
               'expected_token_ids_sha256': canonical_sha(full_ids)}
    return feature, metrics


def strict_parse(text, question_type, allowed):
    # No salvaging unclosed boxes, duplicate labels, outside text, or partial orderings.
    match = re.fullmatch(r'\s*<box>([^<>]+)</box>\s*', text)
    if match is None:
        return None
    labels = [x.strip() for x in match.group(1).split(',')]
    if not labels or len(set(labels)) != len(labels) or any(x not in allowed for x in labels):
        return None
    if question_type == 'ordering' and set(labels) != set(allowed):
        return None
    return labels


def score_one(raw, request, private):
    mapping = request['display_to_content']
    typ = request['question_type']
    labels = strict_parse(raw, typ, mapping)
    pred = [mapping[x] for x in labels] if labels is not None else []
    gold = private['gold_content_ids']
    result = {k: request[k] for k in ('request_id', 'candidate_id', 'root_id', 'variant_id', 'question_type', 'cohort_id')}
    result.update(reporting_role=private['reporting_role'], gold_cardinality=len(gold),
                  parse_valid=labels is not None, prediction_content_ids=pred)
    if typ == 'multi_select':
        p, g = set(pred), set(gold)
        tp, fp, fn = len(p & g), len(p - g), len(g - p)
        result.update(exact=float(labels is not None and p == g), false_positive=fp, false_negative=fn,
                      overselection=float(bool(fp)), option_accuracy=(len(mapping) - fp - fn) / len(mapping) if labels is not None else 0.,
                      partial=tp / len(p | g) if labels is not None and p | g else 0.,
                      f1=2 * tp / (2 * tp + fp + fn) if labels is not None and tp else 0.)
    else:
        positions = {value: i for i, value in enumerate(pred)}
        pairs = [(a, b) for i, a in enumerate(gold) for b in gold[i + 1:]]
        pairwise = sum(positions[a] < positions[b] for a, b in pairs) / len(pairs) if labels is not None else 0.
        result.update(exact=float(labels is not None and pred == gold), pairwise=pairwise)
    return result


def aggregate(rows):
    def group_key(r):
        if r['reporting_role'] == 'GOLD6_CONTROL_ONLY' or (r['question_type'] == 'multi_select' and r['gold_cardinality'] == 6):
            return 'MC6_CONTROL'
        if r['question_type'] == 'ordering': return 'ORDERING'
        if 2 <= r['gold_cardinality'] <= 5: return 'MC2_5_PRIMARY'
        return 'MC_OTHER_CARDINALITY_' + str(r['gold_cardinality'])
    def mean(items, field):
        return sum(x[field] for x in items) / len(items) if items else None
    def stats(items):
        out = {'requests': len(items), 'questions': len({r['candidate_id'] for r in items}), 'parse_valid_rate': mean(items, 'parse_valid'), 'exact': mean(items, 'exact')}
        for field in ('partial', 'f1', 'option_accuracy', 'false_positive', 'false_negative', 'overselection', 'pairwise'):
            subset = [r for r in items if field in r]
            if subset: out[field] = mean(subset, field)
        return out
    groups = collections.defaultdict(list)
    roots = collections.defaultdict(list)
    cardinalities = collections.defaultdict(list)
    cohorts = collections.defaultdict(list)
    by_candidate = collections.defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row); roots[row['root_id']].append(row)
        cardinalities[str(row['gold_cardinality']) if row['question_type'] == 'multi_select' else 'ordering'].append(row)
        cohorts[row['cohort_id']].append(row); by_candidate[row['candidate_id']].append(row)
    consistency = []
    for cid, part in by_candidate.items():
        if len(part) < 2: continue
        values = [tuple(sorted(r['prediction_content_ids'])) if r['question_type'] == 'multi_select' else tuple(r['prediction_content_ids']) for r in part]
        consistency.append(all(r['parse_valid'] for r in part) and len(set(values)) == 1)
    return {'overall': stats(rows), 'by_task': {k: stats(v) for k, v in sorted(groups.items())},
            'by_root': {k: {**stats(v), 'by_task': {task: stats([r for r in v if group_key(r) == task]) for task in sorted({group_key(r) for r in v})}} for k, v in sorted(roots.items())},
            'by_gold_cardinality': {k: stats(v) for k, v in sorted(cardinalities.items())},
            'by_cohort': {k: stats(v) for k, v in sorted(cohorts.items())},
            'display_consistency': {'paired_questions': len(consistency), 'rate': sum(consistency) / len(consistency) if consistency else None}}
