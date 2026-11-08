# Evidence metadata adjudication

The original author targets, first drafts, and first mechanical checks are immutable. Before blind review comparison and before solver evaluation, a separate, source-bound metadata adjudication may correct a transcription error in a citation identifier or origin field without changing the question, options, Gold, quoted text, or semantic target.

A citation identifier correction requires that the original suffix uniquely identifies one actual turn and that every quoted citation using it is an exact substring of that turn. An origin correction requires that the quoted text matches the referenced turn exactly and that the replacement is the source turn's actual origin. The adjudication must retain old and new values, quote-to-turn checks, source hash, original and derived document hashes, and independent source qualification; it must not be used to fill missing evidence or resolve ambiguity. Any unresolved mismatch stays on HOLD. The first mechanical review remains part of the record.

This policy is frozen before blind Gold comparison or any solver call. A corrected evidence identifier does not by itself establish that the cited text supports the option or that the source is qualified.
