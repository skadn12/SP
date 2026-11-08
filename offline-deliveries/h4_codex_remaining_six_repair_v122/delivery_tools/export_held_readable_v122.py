"""Verbatim readable export of held candidates, clearly labeled as unaccepted."""
from pathlib import Path
import json

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    audit = read(B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json')
    manifest = {r['root_id']: r for r in read(B / 'MANIFEST.json')['roots']}
    for row in audit['roots']:
        if row['status'] == 'FINALIZED':
            continue
        root = row['root_id']
        unit = Path(manifest[root]['unit_path'])
        folder = unit / 'roots' / unit.name
        source = folder / 'CUMULATIVE_SOURCE.json'
        if not source.exists():
            latest = sorted(folder.glob('scenes/*/CUMULATIVE_SOURCE.json'))
            if latest:
                source = latest[-1]
            elif manifest[root].get('candidate_prefix_binding'):
                source = Path(manifest[root]['candidate_prefix_binding']['path'])
            else:
                source = unit / 'source_repair/CANDIDATE_BASELINE.json'
        document = read(source)
        lines = [f'# {unit.name}：本轮停止候选的完整可读记录', '',
            f'终态：{row["status"]}。来源类型：{row["source_status"]}。以下内容用于检查失败，未获得训练资格；包含尚未通过或无效审核的内容。', '',
            f'逐字导出来源：[{source.name}]({source})。共{len(document["turns"])}条发言。', '', '## 对话', '']
        for turn in document['turns']:
            lines += [f'**{turn["turn_ref"]} — {turn["speaker"]}**：{turn["text"]}', '']
        public = folder / 'PUBLIC_QUESTIONS.json'
        if public.exists():
            lines += ['## 已生成但未完成有效审核的题目', '', '没有冻结答案；本轮未进行Solver筛选，也未加入训练候选。', '']
            for question in read(public)['questions']:
                lines += [f'### {question["candidate_id"]}', '', question['question'], '']
                for label, proposition in question.get('options', question.get('items', {})).items():
                    lines += [f'- **{label}**：{proposition}']
                lines += ['']
        path = B / 'delivery' / (unit.name + '_DIALOGUE_AND_QUESTIONS.md')
        with path.open('x') as stream:
            stream.write('\n'.join(lines) + '\n')
        print(str(path))


if __name__ == '__main__':
    main()
