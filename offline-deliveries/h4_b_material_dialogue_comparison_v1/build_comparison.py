"""Export existing B material/card/dialogue artifacts; no generation or API calls."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
RUNS = PROJECT / "offline-deliveries/h4_sequential_ab_v51/runs"
POOL = PROJECT / "offline-deliveries/h4_external_event_source_refresh_v1"
INPUT_HASHES: dict[str, str] = {}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def read(path: Path):
    raw = path.read_bytes()
    INPUT_HASHES[str(path)] = sha(raw)
    return json.loads(raw)


def write_json(name: str, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


NOTES = {
    "135466/S01": {
        "original": "查询 Downing College 景点信息，再寻找市中心中等价位欧洲菜餐馆。",
        "expanded": "未生成：未取得可用事件卡。",
        "observations": ["第一轮提取未采用，之后记录提供方请求失败；extractions/01.json 的 card 为 null。"],
    },
    "135466/S02": {
        "original": "Nyasia 首日上学前既兴奋又紧张，母亲鼓励，孩子接受鼓励。",
        "expanded": "Morgan 向 Quinn 表达周末看 State University 橄榄球赛的焦虑，接受朋友鼓励。",
        "observations": [
            "保留“表达复杂情绪 → 对方鼓励 → 接受支持”的三步互动；亲子关系改成朋友关系。",
            "看球、State University、希望独立又能相互支持，均可在 planner 引用的 root 历史中核对。",
            "新对话侧重情绪准备；结尾提出协调出发时间，尚未给出具体出发安排或比赛结果。",
        ],
    },
    "135466/S03": {
        "original": "咨询学院参观信息；Churchill College 的地址、邮编、免费入场；随后转向餐馆和火车。",
        "expanded": "Morgan 向 Quinn 咨询学院参观，获取 Churchill College 的地址、邮编和费用。",
        "observations": [
            "事件卡只保留原文 turn:0–3 的学院信息问答，餐馆/火车转题未进入卡的 steps。",
            "扩充沿用 Churchill College、Storey's Way、db30ds（排版为 DB3 0DS）和免费入场，素材内容保留较多。",
            "人物换成 Morgan/Quinn，并以主动求助接上朋友支持主题；具体任务流程仍接近原始信息问答。",
            "地址和邮编按文件原样展示，此处没有做外部事实校正。",
        ],
    },
    "210046/S01": {
        "original": "寻找市中心中等价位意大利餐馆，随后查询体育活动和博物馆。",
        "expanded": "未生成：未取得可用事件卡。",
        "observations": ["提取记录包含提供方请求失败；extractions/01.json 的 card 为 null。"],
    },
    "210046/S02": {
        "original": "Ulisses 归还四本图书，讨论阅读偏好；图书管理员建议奇幻小说，读者接受。",
        "expanded": "Emily 与 David 处理 Finance 共享目录权限，确定先做 Q3 报表并用评论标记疑问。",
        "observations": [
            "原卡四步为“还书 → 核对归还 → 推荐题材 → 接受推荐”；实际对话只较明显保留“建议 → 接受”的宽泛互动。",
            "实际对话转为入职后的权限处理和工作安排，具体的还书事件及阅读偏好基本消失，素材约束较松。",
            "Planner 将还书类比为职业成长复盘，但实际对话又改为共享盘权限和数据录入；可以直接对照计划与全文。",
        ],
    },
    "210046/S03": {
        "original": "旅游订餐：Bedouin，7 人，周一 17:15 失败，改周一 16:15 成功。",
        "expanded": "项目发布后的团队聚餐：周五 17:15 失败，改周六 16:15 成功。",
        "observations": [
            "保留“餐馆推荐 → 预订 → 无位 → 调整 → 成功”的重试骨架，以及 Bedouin、7 人、17:15、16:15、KCIWXESP。",
            "日期从原文的周一不变改成周五到周六；实际调整了日期和时间。Planner 原计划仍是周一只改时间。",
            "原文 Bedouin 为 expensive，实际对话改成 moderately priced；实际对话还将 Bedouin 描述为中东菜，并保留 Emily 想尝试南非菜的偏好。",
            "人物/聚餐动机接入 David、Emily 的工作关系，任务结构保留清楚；具体属性已有多处改写。",
        ],
    },
}


def complete_units(dataset, native):
    if dataset == "MultiWOZ_2.2":
        return [dict(unit_id=f"turn:{i}", kind="dialogue", speaker=t["speaker"], text=t["utterance"])
                for i, t in enumerate(native["turns"])]
    return [dict(unit_id="narrative", kind="narrative", speaker=None, text=native["narrative"])] + [
        dict(unit_id=f"dialogue:{i}", kind="dialogue", speaker=speaker, text=utterance)
        for i, (speaker, utterance) in enumerate(zip(native["speakers"], native["dialogue"]))]


def check_evidence(card, item):
    units = {u["unit_id"]: u for u in item["units"]}
    for part in card["steps"] + card["dependencies"]:
        for e in part["evidence"]:
            u = units[e["unit_id"]]
            assert u["text"][e["char_start"]:e["char_end"]] == e["quote"]
            assert sha(e["quote"].encode()) == e["quote_sha256"]
            assert u["speaker"] == e["source_speaker"] and u["kind"] == e["kind"]


def make_data():
    full_manifest = read(POOL / "SOURCE_AVAILABILITY_MANIFEST.json")
    assert sha(canonical({k: v for k, v in full_manifest.items() if k != "manifest_sha256"})) == full_manifest["manifest_sha256"]
    pool_index = {i["source_id"]: i for i in full_manifest["items"]}
    multiwoz = read(POOL / "multiwoz_train_sample.json")
    soda = read(POOL / "soda_train_sample.json")
    mw_index = {i["dialogue_id"]: (pos, i) for pos, i in enumerate(multiwoz)}
    soda_index = {str(e["row"]["original_index"]): (pos, e) for pos, e in enumerate(soda["rows"])}
    assert len(mw_index) == len(multiwoz) == 512
    assert len(soda_index) == len(soda["rows"]) == 100
    cases = []
    checks = []
    for root in ["135466", "210046"]:
        base = RUNS / f"root{root}_a3b3_01"
        manifest = read(base / "dataset_sources.json")
        assert sha(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})) == manifest["manifest_sha256"]
        records = read(base / "CANDIDATE_RECORDS.json")
        report = read(base / "CANDIDATE_REPORT.json")
        history = read(base / "CANDIDATE_HISTORY.json")
        baseline = read(base / "FORMAL_BASELINE.json")
        root_retained_turns = 0
        for pos, item in enumerate(manifest["items"], 1):
            scene = f"S{pos:02d}"
            key = f"{root}/{scene}"
            assert item == pool_index[item["source_id"]]
            if item["dataset"] == "MultiWOZ_2.2":
                local_pos, native = mw_index[item["item_id"]]
                pool_file = POOL / "multiwoz_train_sample.json"
                upstream_row_idx = None
            else:
                local_pos, entry = soda_index[item["item_id"]]
                native = entry["row"]
                upstream_row_idx = entry["row_idx"]
                assert upstream_row_idx == item["local_row_index"]
                pool_file = POOL / "soda_train_sample.json"
            assert INPUT_HASHES[str(pool_file)] == item["source_file_sha256"]
            assert sha(canonical(native)) == item["source_item_sha256"]
            assert sha(canonical(item["units"])) == item["excerpt_sha256"]
            full = complete_units(item["dataset"], native)
            full_by_id = {u["unit_id"]: u for u in full}
            assert all(full_by_id[u["unit_id"]] == u for u in item["units"])
            ext_path = base / f"extractions/{pos:02d}.json"
            ext = read(ext_path)
            assert ext["source_id"] == item["source_id"]
            card = ext["card"]
            if card:
                assert card["source_id"] == item["source_id"]
                assert sha(canonical({k: v for k, v in card.items() if k != "card_sha256"})) == card["card_sha256"]
                check_evidence(card, item)
            selected = [r for r in records if r["route"] == "dataset_new_event" and r["scene_id"] == scene]
            retained = [r for r in selected if r["decision"]["artifact_retained"]]
            plans = []
            history_anchors = []
            input_summary = None
            if selected:
                candidate_dir = base / f"candidates/cc-train-episode-{root}/B/{scene}"
                inp_path = candidate_dir / "INPUT.json"
                inp = read(inp_path)
                assert inp["card"] == card
                input_by_ref = {t["turn_ref"]: t for t in inp["history"]}
                input_summary = dict(path=str(inp_path), turns=len(inp["history"]),
                                     formal_baseline_turns=len(baseline["turns"]),
                                     prior_provisional_turns=len(inp["history"]) - len(baseline["turns"]))
                for p in sorted(base.glob(f"requests/*-B-{scene}-planner/RESPONSE.json")):
                    plan = json.loads(read(p)["text"])
                    assert plan["dataset_card_id"] == card["card_id"]
                    plans.append(dict(path=str(p), plan=plan))
                    for anchor in plan["source_anchors"]:
                        ref = anchor["turn_ref"]
                        resolution = "exact_turn_ref"
                        if ref not in input_by_ref:
                            prefix = f"cc-train-episode-{root}:"
                            assert ref.startswith(prefix + "R01:")
                            ref = ref[len(prefix):]
                            resolution = "root_prefix_removed_for_R01_reference"
                        turn = input_by_ref[ref]
                        assert anchor["quote"] in turn["text"]
                        history_anchors.append(dict(anchor=anchor, actual_input_turn=turn, reference_resolution=resolution))
                for rec in selected:
                    attempt = rec["candidate_id"].rsplit("/", 1)[1]
                    saved_record = read(candidate_dir / f"RECORD_{attempt}.json")
                    generated = read(candidate_dir / f"GENERATED_{attempt}.json")
                    assert saved_record == rec
                    assert generated["realization"] == rec["realization"]
                    assert rec["realization"]["dataset_card_id"] == card["card_id"]
            for rec in retained:
                actual = [t for t in history["turns"] if t.get("scene_ref") == f"CANDIDATE:B:{scene}"]
                expected = rec["realization"]["turns"]
                assert [(t["speaker"], t["text"]) for t in actual] == [(t["speaker"], t["text"]) for t in expected]
                root_retained_turns += len(expected)
            if not card:
                assert not selected
                assert any(x["route"] == "dataset_new_event" and x["scene_id"] == scene and
                           x["reason"] == "source_card_unavailable" for x in report["shortfalls"])
            case = dict(root_id=root, scene_id=scene, source_id=item["source_id"],
                        dataset=item["dataset"], source_item=item, original_native_record=native,
                        original_full_units=full, original_pool_array_position=local_pos,
                        soda_upstream_row_idx=upstream_row_idx, extraction_record=ext,
                        card=card, planner_responses=plans, history_anchors=history_anchors,
                        actual_input_history=input_summary, candidate_records=selected,
                        retained_dialogues=[r["realization"] for r in retained],
                        notes=NOTES[key],
                        paths=dict(run_directory=str(base), dataset_sources=str(base / "dataset_sources.json"),
                                   original_pool=str(pool_file), extraction=str(ext_path),
                                   formal_baseline=str(base / "FORMAL_BASELINE.json"),
                                   candidate_history=str(base / "CANDIDATE_HISTORY.json"),
                                   candidate_records=str(base / "CANDIDATE_RECORDS.json"),
                                   candidate_report=str(base / "CANDIDATE_REPORT.json")))
            cases.append(case)
            checks.append(dict(case=key, source_hashes_and_excerpt_match=True,
                               card_hash_and_all_evidence_match=True if card else None,
                               input_card_and_planner_mapping_match=True if selected else None,
                               planner_history_reference_aliases=[dict(planned=a["anchor"]["turn_ref"], actual=a["actual_input_turn"]["turn_ref"])
                                   for a in history_anchors if a["reference_resolution"] != "exact_turn_ref"],
                               full_retained_dialogue_matches_generated_record_and_history=True if retained else None))
        assert root_retained_turns == report["metrics"]["by_route"]["B"]["retained"]["turns"]
    return dict(schema_version="h4.b_material_dialogue_comparison.v1", exported_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                scope="Two specified v51 runs; existing artifacts only", local_source_pool_counts=dict(Counter(i["dataset"] for i in full_manifest["items"])),
                selected_sources=len(cases), successful_cards=sum(bool(c["card"]) for c in cases),
                retained_scenes=sum(len(c["retained_dialogues"]) for c in cases),
                retained_dialogue_turns=sum(len(d["turns"]) for c in cases for d in c["retained_dialogues"]),
                status_note="retained_pending_review candidates; not semantic certification or training admission",
                cases=cases), checks


def transcript(units):
    return "\n\n".join(f"**{u['unit_id']} · {u['speaker'] or '叙述'}**  \n{u['text']}" for u in units)


def markdown(data):
    lines = ["# B 路线原始素材、事件卡与扩充对话对照", "",
             "范围：h4_sequential_ab_v51 中 root135466_a3b3_01 与 root210046_a3b3_01。全部正文从现有文件导出。", "",
             "**共选择 6 条来源；成功提取 4 张事件卡，生成并保留 4 段完整候选对话，共 54 turn。两处 S01 都提取失败。**", "",
             "保留状态为 retained_pending_review，尚待独立复核，未进入训练。下方中文比较为本次阅读观察；英文原文、原卡和对话原样保留。", "",
             "[三栏交互对照](comparison.html) · [四张完整原卡 JSON](source_cards.json) · [全部对照数据 JSON](comparison.json) · [校验结果](VALIDATION.json)", "",
             f"全量本地素材清单：[SOURCE_AVAILABILITY_MANIFEST.json]({POOL / 'SOURCE_AVAILABILITY_MANIFEST.json'})，含 512 条 MultiWOZ 与 100 条 SODA 原文摘录；不代表 612 张已提取事件卡。", "",
             "| Root / B 场景 | 来源 | 卡等级 | 保留对话 |", "|---|---|---|---|"]
    for c in data["cases"]:
        turns = sum(len(x["turns"]) for x in c["retained_dialogues"])
        lines.append(f"| {c['root_id']} / {c['scene_id']} | {c['source_id']} | {c['card']['guidance_level'] if c['card'] else '提取失败'} | {str(turns) + ' turn' if turns else '未生成'} |")
    lines += ["", "阅读顺序：先看实际提取输入，再看结构卡和 root 历史锚点，最后看扩充全文。折叠项另外提供数据集完整原样本，避免把未传入提取器的后续内容算作指导输入。", ""]
    for c in data["cases"]:
        card = c["card"]
        lines += [f"## {c['root_id']} / B / {c['scene_id']} — {c['source_id']}", "",
                  f"原素材：{c['notes']['original']}", "", f"扩充：{c['notes']['expanded']}", "",
                  "本次阅读观察：", ""] + [f"- {n}" for n in c["notes"]["observations"]]
        lines += ["", f"原始文件：[来源摘录]({c['paths']['dataset_sources']}) · [提取记录]({c['paths']['extraction']}) · [保留记录]({c['paths']['candidate_records']})", "",
                  "### 实际送入提取器的原文（完整摘录）", "",
                  f"source_span：`{json.dumps(c['source_item']['source_span'], ensure_ascii=False)}`。原样本 {len(c['original_full_units'])} 个文本单元；提取输入 {len(c['source_item']['units'])} 个文本单元。SODA 单元数包含 narrative。", "",
                  transcript(c["source_item"]["units"]), "", "<details>", "<summary>展开数据集完整原样本（包括未传入提取器的部分）</summary>", "",
                  transcript(c["original_full_units"]), "", "</details>", "", "### 实际采用的事件卡", ""]
        if not card:
            lines += ["没有可用卡片；card 为 null，未生成对应 B 场景。", "", "```json", json.dumps(c["extraction_record"], ensure_ascii=False, indent=2), "```", ""]
            continue
        lines += [f"card_id：`{card['card_id']}`；guidance_level：`{card['guidance_level']}`。", "",
                  "| 步骤 | 原始角色 | 动作 | 状态 | 原文证据单元 |", "|---|---|---|---|---|"]
        actors = {a["actor_id"]: a for a in card["actors"]}
        for s in card["steps"]:
            lines.append(f"| {s['step_id']} | {actors[s['actor_id']]['source_speaker']} | {s['action']} | {s['state']} | {', '.join(e['unit_id'] for e in s['evidence'])} |")
        lines += ["", "<details>", "<summary>完整事件卡 JSON（全部步骤、依赖、证据引文、来源 hash）</summary>", "", "```json", json.dumps(card, ensure_ascii=False, indent=2), "```", "", "</details>", "",
                  "### root 人物与实际输入历史", "",
                  f"生成此场景时输入历史共 {c['actual_input_history']['turns']} turn，其中正式基线 {c['actual_input_history']['formal_baseline_turns']} turn，先前保留候选 {c['actual_input_history']['prior_provisional_turns']} turn。实际历史也包含前面的候选场景，细节见 [INPUT.json]({c['actual_input_history']['path']})。", ""]
        for p in c["planner_responses"]:
            mapping = p["plan"]["actor_mapping"]
            lines += ["角色映射：" + "；".join(f"{actors[m['card_actor_id']]['source_speaker']} → {m['root_actor']}" for m in mapping) + "。", ""]
        for a in c["history_anchors"]:
            t = a["actual_input_turn"]
            alias = f"（计划中写作 `{a['anchor']['turn_ref']}`）" if a['reference_resolution'] != 'exact_turn_ref' else ""
            lines += [f"- `{t['turn_ref']}`{alias} · {t['speaker']} · origin=`{t['origin']}`：{a['anchor']['quote']}"]
        lines += ["", "<details>", "<summary>展开当时的生成计划（与实际生成对话分开）</summary>", ""]
        for p in c["planner_responses"]:
            lines += [f"[原计划文件]({p['path']})", "", "```json", json.dumps(p["plan"], ensure_ascii=False, indent=2), "```", ""]
        lines += ["</details>", "", "### 实际生成并保留的完整对话", ""]
        for r in c["candidate_records"]:
            if not r["decision"]["artifact_retained"]:
                continue
            lines += [f"candidate_id：`{r['candidate_id']}`；状态：`{r['disposition']}`；模型审查 verdict：`{r['decision']['raw_review_verdict']}`。", ""]
            for i, t in enumerate(r["realization"]["turns"], 1):
                lines += [f"**B/{c['scene_id']}/T{i:03d} · {t['speaker']}**  \n{t['text']}", ""]
    return "\n".join(lines)


def html(data):
    template = (OUT / "viewer_template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace("__COMPARISON_DATA__", payload)


def main():
    data, checks = make_data()
    assert (data["selected_sources"], data["successful_cards"], data["retained_scenes"], data["retained_dialogue_turns"]) == (6, 4, 4, 54)
    write_json("comparison.json", data)
    write_json("source_cards.json", dict(scope=data["scope"], successful_cards=4, cards=[
        dict(root_id=c["root_id"], scene_id=c["scene_id"], extraction_path=c["paths"]["extraction"], card=c["card"])
        for c in data["cases"] if c["card"]]))
    write_json("original_selected_items.json", [dict(root_id=c["root_id"], scene_id=c["scene_id"],
        source_item=c["source_item"], original_native_record=c["original_native_record"],
        original_pool_array_position=c["original_pool_array_position"], soda_upstream_row_idx=c["soda_upstream_row_idx"])
        for c in data["cases"]])
    (OUT / "COMPARISON.md").write_text(markdown(data))
    (OUT / "comparison.html").write_text(html(data))
    for path, expected in INPUT_HASHES.items():
        assert sha(Path(path).read_bytes()) == expected
    write_json("VALIDATION.json", dict(status="PASS", selected_sources=6, successful_cards=4,
        retained_scenes=4, retained_turns=54, source_files_unchanged=True,
        checks=checks, input_file_sha256=INPUT_HASHES,
        validation_scope="Exact file/quote/lineage/transcript integrity; qualitative observations are not semantic certification"))
    print(json.dumps(dict(status="PASS", output_directory=str(OUT), selected_sources=6,
        successful_cards=4, retained_scenes=4, retained_turns=54, source_files_unchanged=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
