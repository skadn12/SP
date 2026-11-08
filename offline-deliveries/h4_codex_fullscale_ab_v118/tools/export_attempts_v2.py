"""Add the second retained read-directory relocation to the attempts report.

Preserve the v1 exporter and all attempts. This reporting adapter changes no
source, judgment, question, Gold, Solver result, or selection rule.
"""
import argparse
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL_SHA256 = 'e728cec023d9a2b0c2802933b64dad23a30ca567b887f7e004c0739d66b1fc92'


def engine():
    original = HERE / 'export_attempts_v1.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == ORIGINAL_SHA256
    source = original.read_text()
    old_binding = "                              binding(batch / 'evidence/READ_DIRECTORY_RELOCATION_V1.json'),\n"
    new_binding = old_binding + "                              binding(batch / 'evidence/READ_DIRECTORY_RELOCATION_V2.json'),\n" + "                              binding(batch / 'evidence/AUDIT_RELOCATION_V4_BINDINGS.json'),\n" + "                              binding(batch / 'evidence/ALL_CONTENT_AUDIT_PROCEDURE_ADAPTER.json'),\n" + "                              binding(HERE / 'export_attempts_v1.py'),\n"
    old_line = "             '96462：审者误移动首个失败read目录后新读；原/新判断只差缺失字段补充，原字节均保留。严格relocation审计验证唯一原path+SHA映射，程序偏差仍记录。', '',\n"
    new_line = old_line.replace("', '',\n", "',\n") + "             '112812 B1：同样发生失败read目录迁移，原RAW与SAVE_FAILURE完整归档；新判断仅补全6个已有基线turn引用的root前缀，语义判断未变。v4审计仅对记录的精确原path+SHA解析归档原字节，执行偏差保留。',\n" + "             '112812 B2：保留首次缺少native_event_full_read声明的失败；既有完整原生事件阅读后，以additive声明补充完成机械恢复。所有首稿、失败和恢复收据均保留，不能据此宣称语义认证。', '',\n"
    assert source.count(old_binding) == source.count(old_line) == 1
    source = source.replace(old_binding, new_binding).replace(old_line, new_line)
    ns = {'__file__': str(Path(__file__).resolve()), '__name__': 'retained_attempts_export_v2'}
    exec(compile(source, str(original), 'exec'), ns)
    return ns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    engine()['export'](parser.parse_args().output.resolve())
