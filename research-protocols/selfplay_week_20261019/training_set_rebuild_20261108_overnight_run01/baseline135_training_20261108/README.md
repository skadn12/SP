# Isolated 135-record baseline (prepared)

This directory binds the frozen 135-record M-clean export for the independent
baseline requested by the parallel workflow. It is separate from the final
merged-training directory. The 140-candidate Solver gate may add records to a
later immutable `FINAL_TRAIN.jsonl`; it must not rewrite this baseline input or
its output.

Status: `PREPARED_NOT_LAUNCHED`

The sealed input remains at:

`/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl`

The formal run must use the unchanged native trainer and M-clean starting
adapter recorded in `BASELINE_MANIFEST.json`. The one-record smoke in the
separate `h4_round0_b0_final_merged_training_20261108_A_smoke02` directory is
only a resource check and is not this baseline.

Do not start this directory concurrently with Solver inference on the same GPU.
