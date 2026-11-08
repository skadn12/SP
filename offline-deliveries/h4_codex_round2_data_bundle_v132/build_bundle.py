#!/usr/bin/env python3
"""Copy and index the actual round-two data, without inference or source edits.

The builder owns data/, originals/, DATA_STATISTICS.json, ROOT_INDEX.*, 
SOURCE_FILE_INDEX.json and README_DATA_ZH.md only. Final bundle sealing and
evaluation evidence are deliberately handled separately by the parent task.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
TRAIN = "h4_codex_train_pool_v125"
HELDOUT = "h4_codex_dev_test_ab_v124"
EXECUTION = "h4_codex_training_v126"
PRODUCTION = (
    "h4_codex_fullscale_ab_v118", "h4_codex_scene_retry_v120",
    "h4_codex_all_failed_repair_v121", "h4_codex_remaining_six_repair_v122",
)
EXPECTED_SCENES = tuple(f"R02:{route}:S{n:02d}" for route in "AB" for n in (1, 2, 3))
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def counts(values) -> dict:
    return dict(sorted(collections.Counter(values).items(), key=lambda p: str(p[0])))


class Builder:
    def __init__(self, repo: Path, out: Path):
        self.repo, self.out = repo.resolve(), out.resolve()
        if self.out == self.repo or self.repo not in self.out.parents:
            raise ValueError("Output must be a new delivery directory within the repository")
        self.copies = {}
        self.generated = set()
        self.omissions = []
        self.verified_seals = []

    def source(self, relative: str) -> Path:
        return self.repo / "offline-deliveries" / relative

    def read(self, p: Path):
        return json.loads(p.read_text())

    def write(self, relative: str, data: bytes):
        p = self.out / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() == data:
            return
        # Only explicitly owned derived paths can be replaced; historical copies
        # are handled by copy() and are never overwritten with different bytes.
        p.write_bytes(data)
        self.generated.add(relative)

    def json(self, relative: str, value):
        self.write(relative, encode_json(value))

    def text(self, relative: str, value: str):
        self.write(relative, value.encode())

    def copy(self, src: Path, relative: str, expected: str | None = None):
        src = src.resolve()
        if self.repo not in src.parents:
            raise ValueError(f"External file is not dataset content: {src}")
        if src.suffix.lower() in WEIGHT_SUFFIXES:
            raise ValueError(f"Model weights must not enter the bundle: {src}")
        data = src.read_bytes()
        digest = sha(data)
        if expected and digest != expected:
            raise ValueError(f"Source SHA mismatch: {src}")
        p = self.out / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != data:
            raise ValueError(f"Refusing to replace a different historical copy: {p}")
        if not p.exists():
            p.write_bytes(data)
        binding = {"original_path": str(src), "bundle_path": relative,
                   "sha256": digest, "size_bytes": len(data), "byte_identical": True}
        self.copies[relative] = binding
        return binding

    def original(self, src: Path, expected: str | None = None):
        rel = src.resolve().relative_to(self.repo).as_posix()
        return self.copy(src, "originals/" + rel, expected)

    def tree(self, directory: Path, exclusions=()):
        for p in sorted(directory.rglob("*")):
            rel = p.relative_to(directory)
            if not p.is_file():
                continue
            if "__pycache__" in rel.parts or any(part in exclusions for part in rel.parts):
                self.omissions.append({"path": str(p), "reason": "excluded_runtime_or_bulk_subtree"})
                continue
            if p.suffix.lower() in WEIGHT_SUFFIXES:
                self.omissions.append({"path": str(p), "reason": "model_weights_not_dataset"})
                continue
            if p.is_symlink():
                raise ValueError(f"Dataset snapshot contains a symlink: {p}")
            self.original(p)

    def verify_seal(self, directory: Path):
        manifest = directory / "COMPLETE_MANIFEST.json"
        obj = self.read(manifest)
        checked = 0
        for binding in obj["files"]:
            relative = binding["path"]
            p = Path(relative)
            if not p.is_absolute():
                p = directory / relative
            data = p.read_bytes()
            if sha(data) != binding["sha256"]:
                raise ValueError(f"Immutable source manifest mismatch: {p}")
            size = binding.get("size_bytes", binding.get("bytes"))
            if size is not None and len(data) != size:
                raise ValueError(f"Immutable source manifest size mismatch: {p}")
            checked += 1
        self.verified_seals.append({"manifest_path": str(manifest),
                                    "sha256": sha(manifest.read_bytes()),
                                    "checked_files": checked, "status": "PASS"})

    def snapshots(self):
        for name in (TRAIN, HELDOUT):
            directory = self.source(name)
            self.verify_seal(directory)
            self.tree(directory)
        # Keep original proposals, all attempts, reviews, failures and delivery
        # state. Solver request/response traces are separate evaluation outputs.
        for name in PRODUCTION:
            self.tree(self.source(name), exclusions={"evaluations", "tests"})
        # Preserve each non-weight in-repository input actually frozen by v125,
        # including upstream source/review versions outside the four batches.
        frozen = self.read(self.source(TRAIN + "/INPUT_FREEZE.json"))
        for entry in frozen["files"]:
            p = Path(entry["path"])
            if self.repo not in p.resolve().parents or p.suffix.lower() in WEIGHT_SUFFIXES:
                self.omissions.append({"path": str(p), "reason": "external_model_or_model_weights"})
                continue
            self.original(p, entry["sha256"])
        # These are execution metadata only. No adapter, base weights, or merged
        # checkpoint is copied, including the large smoke adapter directory.
        execution = self.source(EXECUTION)
        for p in sorted(execution.iterdir()):
            if p.is_file():
                self.original(p)
        for p in sorted((execution / "evidence").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                self.original(p)

    def readable(self, split: str, root: str, source: dict, questions: list, state: str):
        lines = [f"# {root}", "", f"划分：{split}；状态：{state}；题目数：{len(questions)}。", "",
                 "本文件仅把已保存的对话、题目和答案排版，未改写原文。", "", "## 完整已物化对话", ""]
        previous = object()
        for turn in source["turns"]:
            scene = turn.get("scene_ref") or turn.get("session_ref") or "parent"
            if scene != previous:
                lines += [f"### {scene}", ""]
                previous = scene
            lines += [f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}", ""]
        lines += ["## 对应的最终题目与答案", ""]
        if not questions:
            lines += ["该 root 没有进入最终题库的题目。原因与未入选尝试见对应原始状态/审核文件。", ""]
        for number, q in enumerate(questions, 1):
            lines += [f"### {number}. {q['candidate_id']}", "", q["question"], ""]
            for key, value in (q.get("items") or q.get("options") or {}).items():
                lines += [f"{key}. {value}", ""]
            gold = q.get("gold") or ([q["correct_option"]] if q.get("correct_option") else [])
            lines += ["Gold：" + ",".join(gold), "", "完整答案、证据和来源字段：", "", "```json",
                      json.dumps(q, ensure_ascii=False, indent=2), "```", ""]
        self.text(f"data/{split}/readable/{root}.md", "\n".join(lines))

    def split_data(self, split: str, qs: list, root_records: list, question_source: Path):
        self.copy(question_source, f"data/{split}/questions_with_answers.json")
        self.text(f"data/{split}/questions_with_answers.jsonl",
                  "".join(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n" for q in qs))
        by_root = collections.defaultdict(list)
        for q in qs:
            by_root[q["root_id"]].append(q)
        records = []
        for record in root_records:
            root = record["root_id"]
            binding = record["source_binding"]
            src_path = Path(binding["path"])
            copied = self.copy(src_path, f"data/{split}/dialogues/{root}.json", binding["sha256"])
            source = self.read(src_path)
            turns = source["turns"]
            scenes = counts(t.get("scene_ref") for t in turns if t.get("scene_ref") in EXPECTED_SCENES)
            rows = by_root[root]
            state = record.get("status", "ACTUALLY_TRAINED")
            if split == "train" and set(scenes) != set(EXPECTED_SCENES):
                raise ValueError(f"Actual training source lacks A/B six scenes: {root}")
            entry = {
                "root_id": root, "split": split, "status": state,
                "usage": "ACTUALLY_TRAINED" if split == "train" else "HELDOUT_FINAL_DELIVERY_ONLY",
                "question_count": len(rows), "question_type_counts": counts(q["question_type"] for q in rows),
                "candidate_ids": [q["candidate_id"] for q in rows],
                "dialogue_path": copied["bundle_path"], "dialogue_sha256": copied["sha256"],
                "question_path": f"data/{split}/questions_by_root/{root}.json",
                "readable_path": f"data/{split}/readable/{root}.md",
                "original_source_path": str(src_path), "total_turns": len(turns),
                "new_r02_turns": sum(scenes.values()), "new_r02_scene_turn_counts": scenes,
                "all_six_scenes_materialized": set(scenes) == set(EXPECTED_SCENES),
                "ready_scene_count": record.get("ready_scene_count", 6 if split == "train" else None),
                "original_root_record": record,
            }
            self.json(entry["question_path"], {"root_id": root, "split": split,
                "usage": entry["usage"], "dialogue_path": entry["dialogue_path"],
                "dialogue_sha256": entry["dialogue_sha256"], "status": state, "questions": rows})
            self.readable(split, root, source, rows, state)
            records.append(entry)
        if set(by_root) - {r["root_id"] for r in root_records}:
            raise ValueError(f"Question root lacks dialogue in {split}")
        self.json(f"data/{split}/ROOT_MANIFEST.json", records)
        path_map = {r["original_source_path"]: r["dialogue_path"] for r in records}
        self.json(f"data/{split}/LOCAL_PATH_MAP.json", path_map)
        return records

    def build(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.snapshots()
        train_dir = self.source(TRAIN)
        questions = self.read(train_dir / "data/COMBINED_QUESTIONS.json")["questions"]
        raw = self.read(train_dir / "data/RAW_SELECTED_QUESTIONS.json")["questions"]
        actual = [json.loads(line) for line in (train_dir / "data/TRAIN_NEW_ONLY.jsonl").read_text().splitlines()]
        train_roots = self.read(train_dir / "data/ROOT_MANIFEST.json")
        assert len(questions) == len(raw) == len(actual) == 363 and len(train_roots) == 68
        assert {q["candidate_id"] for q in questions} == {q["candidate_id"] for q in actual}
        for p in sorted((train_dir / "data").iterdir()):
            if p.is_file():
                self.copy(p, "data/train/actual_train/" + p.name)
        for filename in ("PROTOCOL.json", "TRAINING_PLAN.json", "REPLAY_DISPOSITION.json"):
            self.copy(train_dir / filename, "data/train/actual_train/" + filename)
        self.copy(train_dir / "data/RAW_SELECTED_QUESTIONS.json", "data/train/raw_selected_questions.json")
        execution = self.read(self.source(EXECUTION + "/EXECUTION_FREEZE.json"))
        assert sha((train_dir / "data/TRAIN_NEW_ONLY.jsonl").read_bytes()) == execution["training_data_sha256"]
        roots_index = self.split_data("train", questions, train_roots, train_dir / "data/COMBINED_QUESTIONS.json")
        summary = {}
        for split in ("dev", "test"):
            directory = self.source(HELDOUT + "/delivery/" + split)
            totals = self.read(directory / "TOTALS.json")
            disposition = self.read(directory / "QUESTION_DISPOSITION.json")
            # TEST operations here are mechanical parse, copy, grouping, counts
            # and display only. No semantic judgement, answer revision or model.
            split_questions = self.read(directory / "QUESTIONS_WITH_ANSWERS.json")["questions"]
            source_records = []
            for item in disposition["roots"]:
                root = item["root_id"]
                public = self.read(directory / ("root" + root.rsplit("-", 1)[1] + "_PUBLIC.json"))
                source_records.append({**item, "source_binding": public["source_binding"]})
            records = self.split_data(split, split_questions, source_records, directory / "QUESTIONS_WITH_ANSWERS.json")
            roots_index += records
            for filename in ("TOTALS.json", "QUESTION_DISPOSITION.json"):
                self.copy(directory / filename, "data/" + split + "/" + filename)
            assert len(records) == totals["fixed_roots"]
            assert len(split_questions) == totals["content_pass_questions"]
            summary[split] = {"fixed_roots": len(records), "roots_with_final_questions": sum(bool(r["question_count"]) for r in records),
                "roots_six_scenes_ready": totals["roots_six_scenes_ready"],
                "roots_six_scenes_materialized": sum(r["all_six_scenes_materialized"] for r in records),
                "final_questions": len(split_questions), "question_type_counts": totals["question_types"],
                "multi_select_gold_counts": totals["multi_select_gold_counts"],
                "gold6_control_questions": totals["gold6_control_questions"], "terminal_states": totals["terminal_states"],
                "source_delivery_totals_path": f"data/{split}/TOTALS.json"}
        sets = {s: {r["root_id"] for r in roots_index if r["split"] == s} for s in ("train", "dev", "test")}
        assert not (sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"])
        train_summary = self.read(train_dir / "SUMMARY.json")
        summary["train"] = {"fixed_roots": len(train_roots), "roots_with_final_questions": len(train_roots),
            "roots_six_scenes_materialized": len(train_roots), "actual_training_records": len(actual),
            "final_questions": len(questions), "question_type_counts": counts(q["question_type"] for q in questions),
            "multi_select_gold_counts": counts(len(q["gold"]) for q in questions if q["question_type"] == "multi_select"),
            "replay_questions": 0, "actual_epochs": 2, "preparation_snapshot_status": train_summary["status"],
            "actual_training_data_sha256": execution["training_data_sha256"],
            "input_tokens_historical_v125": train_summary["input_tokens"],
            "dialogue_tokens_historical_v125": train_summary["dialogue_tokens"],
            "token_statistics_note": "Copied historical Qwen-tokenizer statistics; no new tokenization or new TEST inference."}
        self.json("ROOT_INDEX.json", {"schema_version": "v132_root_index_v1", "roots": roots_index})
        fields = ["split", "root_id", "status", "usage", "question_count", "total_turns", "new_r02_turns",
                  "all_six_scenes_materialized", "ready_scene_count", "dialogue_path", "question_path", "readable_path", "dialogue_sha256"]
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(roots_index)
        self.text("ROOT_INDEX.csv", buf.getvalue())
        table = ["# Root 到对话、题目索引", "", "保留全部 68 train、15 DEV、16 TEST root；0 题行同样保留。", "",
                 "| split | root | 状态 | 题数 | 全6场景已物化 | 对话及题目 |", "|---|---|---|---:|---|---|"]
        for r in roots_index:
            table.append(f"| {r['split']} | {r['root_id']} | {r['status']} | {r['question_count']} | {r['all_six_scenes_materialized']} | [可读版]({r['readable_path']}) · [题目JSON]({r['question_path']}) · [对话JSON]({r['dialogue_path']}) |")
        self.text("ROOT_INDEX.md", "\n".join(table) + "\n")
        stats = {"schema_version": "v132_data_statistics_v1", "status": "DATA_COPY_AND_INDEX_COMPLETE_NOT_TOPLEVEL_SEALED",
            "splits": {s: summary[s] for s in ("train", "dev", "test")},
            "total_dialogue_roots": len(roots_index), "total_final_questions": sum(summary[s]["final_questions"] for s in summary),
            "root_split_intersections": {"train_dev": [], "train_test": [], "dev_test": []},
            "new_inference_calls": 0, "test_operations": "Copy, mechanical parse/count/group and readable rendering only; no new Solver or semantic analysis.",
            "source_seals_verified": self.verified_seals}
        self.json("DATA_STATISTICS.json", stats)
        self.readme(summary)
        # This is a data-copy mapping, not the complete package seal. It is
        # deterministic and intentionally excludes concurrent evidence/evals.
        self.json("SOURCE_FILE_INDEX.json", {"schema_version": "v132_source_copy_index_v1",
            "status": "DATA_BRANCH_COPY_INDEX_NOT_TOPLEVEL_COMPLETE_MANIFEST",
            "copy_count": len(self.copies), "copied_bytes": sum(x["size_bytes"] for x in self.copies.values()),
            "source_root": str(self.repo), "copies": sorted(self.copies.values(), key=lambda x: x["bundle_path"]),
            "omissions": sorted(self.omissions, key=lambda x: x["path"]),
            "snapshot_policy": {TRAIN: "Complete byte-identical snapshot excluding __pycache__",
                                HELDOUT: "Complete byte-identical snapshot excluding __pycache__",
                                "production_batches": "All files except evaluations/, tests/, __pycache__ and model weights",
                                "v125_frozen_repo_inputs": "All frozen in-repository non-weight files, with frozen SHA validation",
                                EXECUTION: "Top-level execution metadata and evidence only; smoke adapter/model weights excluded"}})
        return stats

    def readme(self, summary):
        text = f"""# 第二轮实际训练与 DEV / TEST 数据集中副本

本目录整理已存在的数据，没有新增扩充、改题、改答案或模型推理。原仓库文件原位保留；这里提供独立副本。最终总封存清单和压缩包由交付主流程另行生成。

| 划分 | 固定 roots | 有最终题目的 roots | 最终题目 | 多选 | 排序 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| train | 68 | 68 | 363 | 238 | 125 | 第二轮实际 SFT 输入，2 epochs，0 replay |
| DEV | 15 | 7 | 48 | 34 | 14 | 保留验证集，不混入 SFT |
| TEST | 16 | 13 | 87 | 65 | 22 | 保留测试集；本次仅打包复制和计数 |

共 99 个 root、498 道最终题目。DEV / TEST 的多选分别含 7 / 13 道 Gold=6 控制题，详见 DATA_STATISTICS.json。DEV 六场景审核 ready 为 8 个 root，但仅 7 个 root 有最终题目；不能把固定 15 / 16 个 root 当成全部扩充、出题成功。

## 从哪里开始查看

- [ROOT_INDEX.md](ROOT_INDEX.md)：每个 root 到完整对话、题目 JSON、可读版的链接，包含 0 题和失败状态。
- `data/train/dialogues/`、`data/dev/dialogues/`、`data/test/dialogues/`：最终来源 JSON 的逐字节副本。训练 68 个 root 都含历史完整前缀及 A / B 两路线各 3 场景。heldout 全 31 个 root 保留最后物化状态，部分停在 baseline 或中途场景，绝不补造缺失场景。
- 各 split 的 `readable/<root_id>.md`：完整对话后接该 root 的最终题目、答案和原有证据字段。
- 各 split 的 `questions_with_answers.json`：原题库文件逐字节复制；`questions_with_answers.jsonl`、`questions_by_root/`：同一题目对象的分行和分 root 排版。原题干、选项、Gold、证据不变。
- `data/train/actual_train/TRAIN_NEW_ONLY.jsonl`：真正送入第二轮训练的 363 条 messages，保留已经打乱的实际显示顺序与映射后答案；其 SHA 与 v126 执行冻结记录一致。
- `data/train/actual_train/QUESTION_LINEAGE.jsonl`：canonical 与实际训练显示映射、token 计数、来源与 lineage；`raw_selected_questions.json` 与 `actual_train/COMBINED_QUESTIONS.json` 同时保留原始 / effective 版本。
- 各 split 的 `LOCAL_PATH_MAP.json` 与根目录 `SOURCE_FILE_INDEX.json`：原绝对路径到包内副本的映射。原始 JSON 的绝对路径作为历史记录保留；离线阅读应使用本地索引。

## 原始候选、审核和失败记录

`originals/offline-deliveries/{TRAIN}/` 与 `originals/offline-deliveries/{HELDOUT}/` 是完整快照（只去除 Python 缓存）。v124 保留全部 31 个 heldout root 的成功、失败、未入选尝试、审核和终止状态。train 的 v118 / v120 / v121 / v122 原始单位、候选、审核、前缀审计和交付记录也复制到同层对应目录；这些生产批次的重复 Solver 请求 / 回答目录 `evaluations/` 与开发测试脚本 `tests/` 未纳入该 originals 分支。

v125 `INPUT_FREEZE.json` 中仓库内非权重输入也按原相对目录复制到 `originals/`，覆盖更早的 v112 / v113 / v114 / v115 / v117 等来源版本。模型权重、adapter 和仓库外模型文件不属于数据包。逐文件来源、SHA、字节数和省略原因见 SOURCE_FILE_INDEX.json。最终题库与原始候选必须区分：候选 / 审核通过记录不等于实际入训；唯一实际训练输入是上述 TRAIN_NEW_ONLY.jsonl。

## 快照状态与使用边界

v125 中 `READY_FOR_CONTROLLED_TRAINING_NOT_STARTED` / `FROZEN_NOT_TRAINED` 是准备时的不可变状态，现已过时；v126 执行报告确认这 363 条记录已完成 2 epochs。v126 的“尚未 Solver 评估”同样只是当时快照；后续评估由根交付流程单独整理。这里保留各历史文件原文，不回写旧状态。

多选 Gold=2/3/4/5 训练题数分别为 65 / 59 / 57 / 57。排序训练题 125 道均为 6 条目；这些题使用统一的 6 场景模板，每场景一个条目，不能仅凭同模板高分推断复杂时间推理已解决。训练前缀与历史模型训练之间存在 root / 背景复用；本包不是全新未见故事基准。

本次没有新增 TEST 推理或语义分析。TEST 历史暴露情况应以既有协议 / 审核记录为准，不宣称从未查看或完全未触及。可读版只是排版既有文本和答案，不提供新的语义可靠性认证。

## 复现整理

在原仓库运行：

```bash
python offline-deliveries/h4_codex_round2_data_bundle_v132/build_bundle.py --repo-root /root/autodl-tmp/gxy-selfplay
```

脚本先核验 v125 / v124 既有封存 SHA，再复制整理；不调用模型，不复制模型权重，不移动或修改旧文件。重复运行会校验历史副本一致性，仅更新自己负责的派生索引；不操作并发生成的 `evidence/`、根 `evaluations/` 或最终封存文件。
"""
        self.text("README_DATA_ZH.md", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    b = Builder(args.repo_root, args.out)
    summary = b.build()
    print(json.dumps({"status": summary["status"], "roots": summary["total_dialogue_roots"],
                      "questions": summary["total_final_questions"], "copy_count": len(b.copies),
                      "copied_bytes": sum(x["size_bytes"] for x in b.copies.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
