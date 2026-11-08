# Multi-select new-root trial v28

This independent trial targets train roots 231899, 210046, 194259, and 96544. Each root has four initial question slots and at most two later two-slot supplements, for an eight-slot ceiling. A slot may be an abstention. The requested generator is SiliconFlow `Qwen/Qwen3.6-35B-A3B`.

The frozen promoted dialogue (parent plus R01) is the only evidence source. The root's old `candidate_pool.json`, including every question type, is deduplication context only. An unreferenced turn is not proof of a new fact. Drafts require six distinct source turns on one narrow topic, with at least one R01 turn, exact turn references, and contiguous verbatim quotes. Mechanical checks do not establish semantic correctness or novelty; those require content review before any normalized draft is used.

The authoritative split manifest and the candidate-pool SHA must validate before a request is built. `capacity_hints.json` records the screen approval for this initial four-slot trial and the priority scenes; its SHA is bound to each request. `prepare` freezes the source, dedup summary, capacity approval and hints, prompt, request, and hashes. `execute` reserves one API attempt and does not retry. `check-response` runs mechanical checks only. Supplements require an approved `CONTENT_REVIEW.json` from the preceding batch; they are optional and not a quota to fill.

The v25 and v27 inputs and outputs are separate and untouched. This trial does not select training data or run training.
