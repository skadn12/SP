# Source-bound option position binding v1

This independent offline step runs after mechanical source checks and before a new Solver evaluation. It changes display labels only. No model generates or repairs question text, no semantic review is performed, and no historical pool, report, Solver response, or training dataset is overwritten.

## Multiselect

`multiselect.py` accepts a raw mechanically checked pool whose `option_order` is `original_qwen_draft_order`. A source-and-candidate-hash-derived permutation moves each proposition and its complete evidence record together. Gold is recomputed from the mapped `supports` relations. For two through five true options, the canonical Gold set must not be a leading prefix. Six-true questions remain permitted and necessarily retain `ABCDEF`; they are reported separately. The output passes the existing multiselect Solver v2 pool contract.

```bash
.venv-local4b/bin/python offline-deliveries/h4_option_position_binding_v1/multiselect.py build \
  --input-pool offline-deliveries/h4_crossroot_multiselect_batch_v32_raw_newroots/solver_pool/POOL.json \
  --output-dir offline-deliveries/h4_option_position_binding_v1/v32_multiselect_rebound
.venv-local4b/bin/python offline-deliveries/h4_option_position_binding_v1/multiselect.py verify \
  --input-pool offline-deliveries/h4_crossroot_multiselect_batch_v32_raw_newroots/solver_pool/POOL.json \
  --output-dir offline-deliveries/h4_option_position_binding_v1/v32_multiselect_rebound
```

The frozen v32 replay has 15 rebound questions: 12/12 non-six questions have non-prefix Gold, with nine distinct non-six masks; three questions are six-true. Original Qwen texts, question wording, evidence, and truth relations are preserved. `POSITION_BINDING.json` records the exact displayed-label-to-original-label map and input hash. A new Solver run is required: v32 responses and its 9 policy selections are tied to the old canonical display and cannot be reused for the rebound pool.

## Ordering

`ordering.py` accepts six-item conversation-order questions and their frozen root `source.json` files. It first checks original Gold against source turn order and rejects bad Gold. It then chooses a deterministic nontrivial label order, moves every item with its evidence, recomputes Gold and `order_reason` from source turn order, and validates the result. Each output Gold has at least four displaced labels, does not begin with `I1` or end with `I6`, and uses six distinct anchor turns. It rejects duplicate six-turn sets within a root.

```bash
.venv-local4b/bin/python offline-deliveries/h4_option_position_binding_v1/ordering.py \
  --input-pool offline-deliveries/h4_crossroot_ordering_no_codex_pilot_v1/LABEL_REBOUND_UNREVIEWED.json \
  --source-root offline-deliveries/h4_crossroot_ordering_batch_v5_fresh_roots \
  --output-dir offline-deliveries/h4_option_position_binding_v1/v5_ordering_rebound
.venv-local4b/bin/python offline-deliveries/h4_option_position_binding_v1/ordering.py \
  --input-pool offline-deliveries/h4_crossroot_ordering_no_codex_pilot_v1/LABEL_REBOUND_UNREVIEWED.json \
  --source-root offline-deliveries/h4_crossroot_ordering_batch_v5_fresh_roots \
  --verify-dir offline-deliveries/h4_option_position_binding_v1/v5_ordering_rebound
```

The frozen replay bound 10/10 ordering candidates with ten distinct Gold orders. This pool is mechanical-only. The existing ordering Solver runner is bound to a fixed reviewed pool, so a normal review and a versioned Solver adapter remain necessary before evaluation; these questions are not newly Solver-selected.

Run focused checks with `.venv-local4b/bin/python -m unittest discover -s offline-deliveries/h4_option_position_binding_v1 -p 'test_*.py'`.
