# Named Input Cleaning Batch v1

This package is an offline preprocessing boundary for expansion inputs. It
does not import or call Planner, Realizer, QA Writer, Memorax, Solver, or any
external API. The source files are read-only; output is written to a new
directory.

## Audit command

The ten-record alias stability audit is reproducible with:

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.named_input_cleaning_v1.batch \
  --manifest scripts/current_method/restart_n5_spell/named_input_cleaning_v1/audit10_alias_manifest.json \
  --output-root /root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/named_batch_v1_audit10_alias_v7_final
```

The final successful audit artifact is:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/named_batch_v1_audit10_alias_v7_final
```

The independent repeat is preserved at:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/named_batch_v1_audit10_alias_v8_repeat
```

The expansion input glob is:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/named_batch_v1_audit10_alias_v7_final/roots/*/unified_root.json
```

## Guarantees and limits

The manifest supplies an explicit two-name mapping per root. Full `Neighbors
A`, `Neighbor A`, `Classmates B`, and `Co-workers A` variants are replaced.
The unambiguous pair forms `Classmates A and B` and bare `A and B` are also
replaced. An isolated bare `A`/`B` is replaced only when punctuation makes it a
direct address, such as `Thanks, B.` or `Hey, A,`; ordinary article phrases such
as `A bakery` and `A few days` are preserved. Common subject-verb forms such as
`A said ...` are reported as unresolved instead of being silently kept. Any
remaining ambiguous standalone short token fails the record rather than being
guessed.

Each record is transformed twice and compared byte-for-byte after canonical
JSON serialization. The report also checks source immutability, unique names,
turn/session counts and order, source provenance, residual aliases, and the
existing `unified_root` reader contract. Only a passed root should be handed
to an expansion runner.

The current audit selected ten raw Conversation Chronicles records:

```text
episode-178392, episode-91716, episode-13836, episode-146700, episode-34406,
episode-12596, episode-216742, episode-41625, episode-96974, episode-105961
```
