# Realizer Prompt Alignment v3

The incremental Realizer now shares the applicable writing instructions and local style extraction with the builder used by the successful `B_TEXT_FREE` trial. The reference is the actual sent request at `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_realizer_ab_single_generation_v1_live_1/b_text_free/actual_messages.json`; its recorded result contains 3 scenes, 44 turns, a passed closure check, and no exact copied turns. That observation does not establish prompt causality or semantic correctness.

| Successful Request Component | Current Integration |
| --- | --- |
| One output scene per input, same order, matching JSON example | Dynamic N=1/2, N scene objects, independent scene-count and turn-count self-checks |
| Opening, interaction, transition, immediate consequence, closing | Shared `realizer_arc_turn_rules(actual_progress_only=True)`; arc follows actual progress, even when the final goal remains partial |
| Continue after the main action with a concrete reaction, clarification, decision or consequence | Same shared writing block; no early stop or unrelated filler |
| Default 14 turns, extend to 15-18 when needed, hard 9-48 | Shared turn instructions and per-scene self-check; no added speakers merely to satisfy a count |
| Required participant coverage, stable identities, auxiliary speakers | Shared `REALIZER_PARTICIPANT_RULES`; current field aliases and speaking coverage are explicit |
| Continuity tail and style profile without historical wording | Shared `build_realizer_style_context`, computed locally from authoritative/retained turns; no arbitrary nested caller data accepted |
| No reconstructing omitted history or repeating emotional endings | Shared `REALIZER_TEXT_FREE_STYLE_RULES` in text-free mode |
| New facts/relations, distinguishable states, natural development, no QA leakage | Adapted `REALIZER_CONTENT_GUIDANCE_V1`; frozen Planner choices remain binding |
| Later scenes follow actual prior dialogue | Retained prefix and actual earlier scenes in this output batch; rejected output and planned after-state are not facts |

The successful trial's fixed three-scene count, treatment of Memory summaries as established facts, compulsory final-goal completion, and free choice of new events are incompatible with the current incremental policy. They are not imported. `TARGET_RULES`, actor ownership, current output schema, batch limits, and existing admission checks continue to apply.

Current route: `workflow.execute_policy` calls `prompts.render_realizer` during preflight and immediately before transmission. The final messages are counted by the existing capacity checks. Prompt versions are `incremental_retention_v1.realizer.successful_prompt_alignment_v3.text_free` and `.full_parent`.

The legacy builder also uses the shared blocks with its original default semantics; its rendered text is checked against the preceding source snapshot. Historical run artifacts are not rewritten. This alignment is verified offline; it does not itself schedule another real model request.
