# Relation Miner Joint Exemplars v1

These synthetic examples demonstrate relation discovery only. They contain no facts,
people, events, or wording from the target dialogue. Do not copy their surface story.

## Positive 1: background_requirement_matching

Parent evidence:

Elena spent three years translating hospital discharge instructions and medical consent forms.

Child evidence:

A community clinic later announces a volunteer position requiring at least three years of experience translating medical documents.

Parent-only:

The reader knows Elena's experience but does not know the later position or its requirement.

Child-only:

The reader knows the requirement but does not know whether Elena satisfies it.

Combined conclusion:

Elena satisfies the clinic position's three-year medical-document translation requirement.

## Positive 2: prior_evidence_later_choice

Parent evidence:

Omar reports that introducing a free weekend shuttle increased attendance among older residents at a health center.

Child evidence:

The town later considers either distributing more flyers or operating a weekend shuttle to improve attendance at a new clinic.

Parent-only:

The reader knows the earlier result but not the later alternatives.

Child-only:

The reader knows the alternatives but has no evidence showing which one is more likely to work.

Combined conclusion:

Operating the weekend shuttle is the later proposal supported by Omar's earlier observed result.

## Positive 3: source_verification_synthesis

Parent evidence:

Lena reports that playground lights repeatedly stop working after heavy rain.

Child evidence:

An electrician later finds water entering a damaged junction box, while a maintenance log confirms that the outages occurred after storms.

Parent-only:

The reader has an unverified complaint but not its later confirmation or explanation.

Child-only:

The reader has later technical evidence but cannot identify which earlier report it verifies.

Combined conclusion:

Lena's report of rain-related light failures was verified, and water entering the damaged junction box explained the failures.

## Negative: child_already_contains_complete_conclusion

Parent evidence:

David has three years of community-service experience.

Child evidence:

The hiring coordinator says, "David qualifies for this position because his three years of community service meet its experience requirement."

Reject:

The child evidence already states the complete qualification conclusion. Parent evidence is unnecessary, so this is not a joint relation.

## Generalization rule

The combined conclusion must state a concrete result that becomes uniquely available
only after combining both sides. Topic overlap is insufficient. Do not accept vague
statements such as "A aligns with B," "A provides a foundation for B," or "A and B
both emphasize X."
